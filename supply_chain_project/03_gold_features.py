# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *
import math

# COMMAND ----------


CATALOG = "supply_chain_catalog"
SILVER_SCHEMA = "silver_schema"
GOLD_SCHEMA = "gold_schema"

SILVER_SALES = f"{CATALOG}.{SILVER_SCHEMA}.cleaned_sales"
SILVER_STORE = f"{CATALOG}.{SILVER_SCHEMA}.cleaned_stores"
GOLD_FEATURES = f"{CATALOG}.{GOLD_SCHEMA}.demand_features"


# COMMAND ----------


class SupplyChainFeatureEngineer:
    
    # Comprehensive feature engineering for supply chain demand forecasting
    
    # Creates 30+ features across multiple categories:
    # - Temporal features (time-based patterns)
    # - Lag features (historical demand)
    # - Rolling window features (trends and volatility)
    # - Growth features (momentum indicators)
    # - Cyclic features (seasonal encoding)
    # - Supply chain specific features (business domain)
    
    
    def __init__(self, sales_df, store_df):
        self.sales_df = sales_df
        self.store_df = store_df
        self.features_df = None
        
    def merge_sales_and_store_data(self):
        """Step 1: Join sales and store data"""
        
        print("STEP 1: MERGING SALES AND STORE DATA")
        
        sales_cols = set(self.sales_df.columns)
        store_cols = set(self.store_df.columns)

        common_cols = sales_cols & store_cols - {"Store"}

        store_df_renamed = self.store_df
        for c in common_cols:
            store_df_renamed = store_df_renamed.withColumnRenamed(c, f"store_{c}")

        self.features_df = self.sales_df.join(
            store_df_renamed,
            on="Store",
            how="left"
        )

        # self.features_df = self.sales_df.join(
        #     self.store_df,
        #     on="Store",
        #     how="left"
        # )
        
        merged_count = self.features_df.count()
        print(f"Merged data: {merged_count:,} records")
        print(f"Columns: {len(self.features_df.columns)}")
        
        return self
    
    def add_temporal_features(self):
        """Step 2: Add time-based features"""
        
        print("STEP 2: CREATING TEMPORAL FEATURES")
        
        
        df = self.features_df
        
        # Basic temporal extractions
        df = df \
            .withColumn("year", year("Date")) \
            .withColumn("month", month("Date")) \
            .withColumn("day", dayofmonth("Date")) \
            .withColumn("day_of_week", dayofweek("Date")) \
            .withColumn("week_of_year", weekofyear("Date")) \
            .withColumn("quarter", quarter("Date"))
        
        print("Added: year, month, day, day_of_week, week_of_year, quarter")
        
        # Derived temporal features
        df = df \
            .withColumn("is_weekend",
                when(col("day_of_week").isin([1, 7]), 1).otherwise(0)
            ) \
            .withColumn("is_month_start",
                when(col("day") <= 7, 1).otherwise(0)
            ) \
            .withColumn("is_month_end",
                when(col("day") >= 25, 1).otherwise(0)
            ) \
            .withColumn("days_in_month",
                dayofmonth(last_day("Date"))
            )
        
        print("Added: is_weekend, is_month_start, is_month_end, days_in_month")
        
        # Season encoding
        df = df.withColumn("season",
            when((col("month") >= 3) & (col("month") <= 5), "Spring")
            .when((col("month") >= 6) & (col("month") <= 8), "Summer")
            .when((col("month") >= 9) & (col("month") <= 11), "Fall")
            .otherwise("Winter")
        )
        
        print("Added: season")
        
        # Encode season as numeric
        season_mapping = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
        df = df.withColumn("season_num",
            when(col("season") == "Spring", 1)
            .when(col("season") == "Summer", 2)
            .when(col("season") == "Fall", 3)
            .otherwise(4)
        )
        
        print("Added: season_num")
        
        self.features_df = df
        print(f"Total temporal features: 13")
        
        return self
    
    def add_lag_features(self):
        """Step 3: Add lagged sales features"""
        
        print("STEP 3: CREATING LAG FEATURES")
        
        
        # Window specification - partition by Store, order by Date
        window_spec = Window.partitionBy("Store").orderBy("Date")
        
        df = self.features_df
        
        # Lag features at different time intervals
        lags = [1, 7, 14, 21, 30, 60, 90, 365]
        
        for lag_days in lags:
            df = df.withColumn(
                f"sales_lag_{lag_days}",
                lag("Sales", lag_days).over(window_spec)
            )
            print(f"Created: sales_lag_{lag_days}")
        
        # Customer lag features
        df = df \
            .withColumn("customers_lag_7", lag("Customers", 7).over(window_spec)) \
            .withColumn("customers_lag_30", lag("Customers", 30).over(window_spec))
        
        print(f"Created: customers_lag_7, customers_lag_30")
        
        self.features_df = df
        print(f"Total lag features: 10")
        
        return self
    
    def add_rolling_window_features(self):
        """Step 4: Add rolling window statistics"""
        
        print("STEP 4: CREATING ROLLING WINDOW FEATURES")
        
        
        # Define rolling windows
        window_7d = Window.partitionBy("Store").orderBy("Date").rowsBetween(-7, -1)
        window_14d = Window.partitionBy("Store").orderBy("Date").rowsBetween(-14, -1)
        window_30d = Window.partitionBy("Store").orderBy("Date").rowsBetween(-30, -1)
        window_60d = Window.partitionBy("Store").orderBy("Date").rowsBetween(-60, -1)
        window_90d = Window.partitionBy("Store").orderBy("Date").rowsBetween(-90, -1)
        
        df = self.features_df
        
        # Rolling means (moving averages)
        df = df \
            .withColumn("rolling_mean_7", avg("Sales").over(window_7d)) \
            .withColumn("rolling_mean_14", avg("Sales").over(window_14d)) \
            .withColumn("rolling_mean_30", avg("Sales").over(window_30d)) \
            .withColumn("rolling_mean_60", avg("Sales").over(window_60d)) \
            .withColumn("rolling_mean_90", avg("Sales").over(window_90d))
        
        print("Created: rolling_mean_7, 14, 30, 60, 90")
        
        # Rolling standard deviations
        df = df \
            .withColumn("rolling_std_7", stddev("Sales").over(window_7d)) \
            .withColumn("rolling_std_30", stddev("Sales").over(window_30d)) \
            .withColumn("rolling_std_90", stddev("Sales").over(window_90d))
        
        print("Created: rolling_std_7, 30, 90")
        
        # Rolling min/max
        df = df \
            .withColumn("rolling_min_30", min("Sales").over(window_30d)) \
            .withColumn("rolling_max_30", max("Sales").over(window_30d)) \
            .withColumn("rolling_median_30", expr("percentile_approx(Sales, 0.5)").over(window_30d))
        
        print("Created: rolling_min_30, rolling_max_30, rolling_median_30")
        
        # Rolling customer metrics
        df = df \
            .withColumn("rolling_customers_mean_30", avg("Customers").over(window_30d))
        
        print("Created: rolling_customers_mean_30")
        
        self.features_df = df
        print(f"Total rolling window features: 12")
        
        return self
    
    def add_growth_and_trend_features(self):
        """Step 5: Add growth rates and trend indicators"""
        
        print("STEP 5: CREATING GROWTH & TREND FEATURES")
        
        
        df = self.features_df
        
        # Week-over-week growth
        df = df.withColumn("sales_growth_7d",
            when(col("sales_lag_7") > 0,
                ((col("Sales") - col("sales_lag_7")) / col("sales_lag_7")) * 100
            ).otherwise(0)
        )
        print("Created: sales_growth_7d")
        
        # Month-over-month growth
        df = df.withColumn("sales_growth_30d",
            when(col("sales_lag_30") > 0,
                ((col("Sales") - col("sales_lag_30")) / col("sales_lag_30")) * 100
            ).otherwise(0)
        )
        print("Created: sales_growth_30d")
        
        # Year-over-year growth
        df = df.withColumn("sales_growth_365d",
            when(col("sales_lag_365") > 0,
                ((col("Sales") - col("sales_lag_365")) / col("sales_lag_365")) * 100
            ).otherwise(0)
        )
        print("Created: sales_growth_365d")
        
        # Demand volatility (coefficient of variation)
        df = df.withColumn("demand_volatility",
            when(col("rolling_mean_30") > 0,
                col("rolling_std_30") / col("rolling_mean_30")
            ).otherwise(0)
        )
        print("Created: demand_volatility")
        
        # Sales momentum (short-term vs long-term trend)
        df = df.withColumn("sales_momentum",
            when((col("rolling_mean_7") > 0) & (col("rolling_mean_30") > 0),
                col("rolling_mean_7") / col("rolling_mean_30")
            ).otherwise(1)
        )
        print("Created: sales_momentum")
        
        # Trend direction
        df = df.withColumn("trend_direction",
            when(col("sales_momentum") > 1.1, "Increasing")
            .when(col("sales_momentum") < 0.9, "Decreasing")
            .otherwise("Stable")
        )
        print("Created: trend_direction")
        
        self.features_df = df
        print(f"Total growth/trend features: 6")
        
        return self
    
    def add_cyclic_encoding_features(self):
        """Step 6: Add sine/cosine encoding for cyclic features"""
        
        print("STEP 6: CREATING CYCLIC ENCODING FEATURES")
        
        
        df = self.features_df
        
        # Encode day of week (1-7) as sine/cosine
        df = df \
            .withColumn("day_of_week_sin",
                sin(col("day_of_week") * (2 * math.pi / 7))
            ) \
            .withColumn("day_of_week_cos",
                cos(col("day_of_week") * (2 * math.pi / 7))
            )
        print("Created: day_of_week_sin, day_of_week_cos")
        
        # Encode month (1-12) as sine/cosine
        df = df \
            .withColumn("month_sin",
                sin(col("month") * (2 * math.pi / 12))
            ) \
            .withColumn("month_cos",
                cos(col("month") * (2 * math.pi / 12))
            )
        print("Created: month_sin, month_cos")
        
        # Encode day of month (1-31) as sine/cosine
        df = df \
            .withColumn("day_sin",
                sin(col("day") * (2 * math.pi / 31))
            ) \
            .withColumn("day_cos",
                cos(col("day") * (2 * math.pi / 31))
            )
        print("Created: day_sin, day_cos")
        
        self.features_df = df
        print(f"Total cyclic encoding features: 6")
        
        return self
    
    def add_supply_chain_domain_features(self):
        """Step 7: Add custom supply chain business features"""
        
        print("STEP 7: CREATING SUPPLY CHAIN DOMAIN FEATURES")
        
        
        df = self.features_df
        
        # Competition pressure score (higher = more competition)
        df = df.withColumn("competition_pressure",
            when(col("CompetitionDistance") < 500, 5)      # Very high pressure
            .when(col("CompetitionDistance") < 1000, 4)    # High pressure
            .when(col("CompetitionDistance") < 3000, 3)    # Medium pressure
            .when(col("CompetitionDistance") < 10000, 2)   # Low pressure
            .when(col("CompetitionDistance") < 999999, 1)  # Very low pressure
            .otherwise(0)                                   # No competition
        )
        print("Created: competition_pressure")
        
        # Store performance tier (based on average sales)
        window_store = Window.partitionBy("Store")
        df = df.withColumn("avg_sales_per_store",
            avg("Sales").over(window_store)
        )
        
        df = df.withColumn("store_performance_tier",
            when(col("avg_sales_per_store") > 8000, "High")
            .when(col("avg_sales_per_store") > 5000, "Medium")
            .otherwise("Low")
        )
        print("Created: avg_sales_per_store, store_performance_tier")
        
        # Sales per customer (efficiency metric)
        df = df.withColumn("sales_per_customer",
            when(col("Customers") > 0,
                col("Sales") / col("Customers")
            ).otherwise(0)
        )
        print("Created: sales_per_customer")
        
        # Promotional effectiveness
        df = df.withColumn("promo_effectiveness",
            when((col("Promo") == 1) & (col("Sales") > col("rolling_mean_30")), "Effective")
            .when((col("Promo") == 1) & (col("Sales") <= col("rolling_mean_30")), "Ineffective")
            .otherwise("No Promo")
        )
        print("Created: promo_effectiveness")
        
        # Promo frequency (rolling 30-day window)
        window_30d = Window.partitionBy("Store").orderBy("Date").rowsBetween(-30, -1)
        df = df.withColumn("promo_frequency_30d",
            avg("Promo").over(window_30d)
        )
        print("Created: promo_frequency_30d")
        
        # Stockout risk indicator
        df = df.withColumn("potential_stockout",
            when((col("Open") == 1) & 
                 (col("Sales") < 100) & 
                 (col("Customers") > 50), 1)
            .otherwise(0)
        )
        print("Created: potential_stockout")
        
        # Holiday impact multiplier
        df = df.withColumn("holiday_sales_multiplier",
            when(col("is_holiday") == 1,
                when(col("rolling_mean_30") > 0,
                    col("Sales") / col("rolling_mean_30")
                ).otherwise(1)
            ).otherwise(1)
        )
        print("Created: holiday_sales_multiplier")
        
        # Store assortment encoding
        df = df.withColumn("assortment_score",
            when(col("Assortment") == "A", 1)  # Basic
            .when(col("Assortment") == "B", 2)  # Extra
            .when(col("Assortment") == "C", 3)  # Extended
            .otherwise(1)
        )
        print("Created: assortment_score")
        
        # Store type encoding
        df = df.withColumn("store_type_score",
            when(col("StoreType") == "A", 1)
            .when(col("StoreType") == "B", 2)
            .when(col("StoreType") == "C", 3)
            .when(col("StoreType") == "D", 4)
            .otherwise(1)
        )
        print("Created: store_type_score")
        
        self.features_df = df
        print(f"Total supply chain domain features: 11")
        
        return self
    
    def cleanup_and_finalize(self):
        """Step 8: Remove rows with nulls and add metadata"""
        
        print("STEP 8: CLEANUP AND FINALIZATION")
        
        
        df = self.features_df
        
        before_count = df.count()
        print(f"Records before cleanup: {before_count:,}")
        
        # Remove rows where critical lag features are null
        # (This happens for the first N days of each store)
        df = df.filter(
            col("sales_lag_7").isNotNull() &
            col("sales_lag_30").isNotNull() &
            col("rolling_mean_7").isNotNull() &
            col("rolling_mean_30").isNotNull()
        )
        
        after_count = df.count()
        removed = before_count - after_count
        retention_rate = (after_count / before_count) * 100
        
        print(f"Records after cleanup: {after_count:,}")
        print(f"Removed records: {removed:,}")
        print(f"Retention rate: {retention_rate:.2f}%")
        
        # Add feature engineering metadata
        df = df \
            .withColumn("feature_engineering_timestamp", current_timestamp()) \
            .withColumn("feature_version", lit("v1.0")) \
            .withColumn("total_features", lit(58))  # Update based on actual count
        
        print("Added metadata columns")
        
        self.features_df = df
        
        return self
    
    def get_feature_summary(self):
        """Generate comprehensive feature summary"""
        
        print("FEATURE ENGINEERING SUMMARY")
        
        
        feature_categories = {
            "Temporal Features": [
                "year", "month", "day", "day_of_week", "week_of_year", "quarter",
                "is_weekend", "is_month_start", "is_month_end", "days_in_month",
                "season", "season_num"
            ],
            "Lag Features": [
                "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_21",
                "sales_lag_30", "sales_lag_60", "sales_lag_90", "sales_lag_365",
                "customers_lag_7", "customers_lag_30"
            ],
            "Rolling Window Features": [
                "rolling_mean_7", "rolling_mean_14", "rolling_mean_30", "rolling_mean_60", "rolling_mean_90",
                "rolling_std_7", "rolling_std_30", "rolling_std_90",
                "rolling_min_30", "rolling_max_30", "rolling_median_30",
                "rolling_customers_mean_30"
            ],
            "Growth & Trend Features": [
                "sales_growth_7d", "sales_growth_30d", "sales_growth_365d",
                "demand_volatility", "sales_momentum", "trend_direction"
            ],
            "Cyclic Encoding Features": [
                "day_of_week_sin", "day_of_week_cos",
                "month_sin", "month_cos",
                "day_sin", "day_cos"
            ],
            "Supply Chain Domain Features": [
                "competition_pressure", "avg_sales_per_store", "store_performance_tier",
                "sales_per_customer", "promo_effectiveness", "promo_frequency_30d",
                "potential_stockout", "holiday_sales_multiplier",
                "assortment_score", "store_type_score"
            ],
            "Original Store Features": [
                "StoreType", "Assortment", "CompetitionDistance", "Promo2",
                "has_competition", "competition_proximity"
            ],
            "Original Sales Features": [
                "Open", "Promo", "StateHoliday", "SchoolHoliday",
                "is_holiday", "is_promo_day"
            ]
        }
        
        total_features = 0
        for category, features in feature_categories.items():
            count = len(features)
            total_features += count
            print(f"\n{category}: {count} features")
            # Print first 5 features as examples
            examples = features[:5]
            if len(features) > 5:
                print(f"  Examples: {', '.join(examples)}...")
            else:
                print(f"  Features: {', '.join(features)}")
        
        print(f"\n{'='*80}")
        print(f"TOTAL FEATURES CREATED: {total_features}")
        print(f"{'='*80}")
        
        return feature_categories


# COMMAND ----------

print("Loading Silver layer data...")
silver_sales = spark.table(SILVER_SALES)
silver_store = spark.table(SILVER_STORE)

print(f"Loaded {silver_sales.count():,} sales records")
print(f"Loaded {silver_store.count():,} store records")


# COMMAND ----------

engineer = SupplyChainFeatureEngineer(silver_sales, silver_store)


# COMMAND ----------

print("Executing feature engineering pipeline...")

features_df = (engineer
    .merge_sales_and_store_data()
    .add_temporal_features()
    .add_lag_features()
    .add_rolling_window_features()
    .add_growth_and_trend_features()
    .add_cyclic_encoding_features()
    .add_supply_chain_domain_features()
    .cleanup_and_finalize()
    .features_df
)


# COMMAND ----------

# Get feature summary
feature_categories = engineer.get_feature_summary()


# COMMAND ----------

features_df.printSchema()

# COMMAND ----------


print("WRITING TO GOLD LAYER")


print(f" Writing features to {GOLD_FEATURES}...")

features_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("mergeSchema", "true") \
    .saveAsTable(GOLD_FEATURES)

# Set table properties
spark.sql(f"""
    ALTER TABLE {GOLD_FEATURES}
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true',
        'quality.layer' = 'gold',
        'feature.version' = 'v1.0',
        'total.features' = '58'
    )
""")

# Add comprehensive comment
spark.sql(f"""
    COMMENT ON TABLE {GOLD_FEATURES} IS
    'Feature-engineered dataset for demand forecasting ML model.
    Contains 58 features across 8 categories: temporal, lag, rolling window,
    growth/trend, cyclic encoding, and supply chain domain features.
    Ready for ML model training.
    Source: {SILVER_SALES} joined with {SILVER_STORE}'
""")


# COMMAND ----------

final_count = spark.table(GOLD_FEATURES).count()
print(f"Successfully created {GOLD_FEATURES}")
print(f"  Records: {final_count:,}")
print(f"  Features: 58")


# COMMAND ----------

print("OPTIMIZING GOLD TABLE")

print(f"Optimizing {GOLD_FEATURES}...")
spark.sql(f"OPTIMIZE {GOLD_FEATURES}")
spark.sql(f"ANALYZE TABLE {GOLD_FEATURES} COMPUTE STATISTICS FOR ALL COLUMNS")
print("Optimization complete")


# COMMAND ----------


print("FEATURE CORRELATION ANALYSIS")


# Select numeric features for correlation
numeric_features = [
    "Sales", "sales_lag_7", "sales_lag_30", "rolling_mean_30",
    "demand_volatility", "sales_momentum", "sales_per_customer",
    "competition_pressure", "promo_frequency_30d"
]

print(f"\nCalculating correlations for key features...")
feature_pdf = spark.table(GOLD_FEATURES).select(numeric_features).toPandas()


# COMMAND ----------


import pandas as pd
correlation_matrix = feature_pdf.corr()

print("\nTop 5 Features Correlated with Sales:")
sales_corr = correlation_matrix["Sales"].sort_values(ascending=False)[1:6]
for feature, corr in sales_corr.items():
    print(f"  {feature}: {corr:.3f}")

# Save correlation analysis
corr_df = spark.createDataFrame(
    [(feat, float(corr)) for feat, corr in sales_corr.items()],
    ["feature", "correlation_with_sales"]
)
corr_df.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.feature_correlation_analysis")

print("Saved correlation analysis")


# COMMAND ----------


print("CREATING FEATURE DICTIONARY")


feature_dict_data = []
for category, features in feature_categories.items():
    for feature in features:
        feature_dict_data.append((
            feature,
            category,
            "Numeric" if feature not in ["StoreType", "Assortment", "StateHoliday", 
                                         "season", "trend_direction", "promo_effectiveness",
                                         "store_performance_tier", "competition_proximity"] else "Categorical",
            "Feature description here"  # You can add specific descriptions
        ))

feature_dict_df = spark.createDataFrame(
    feature_dict_data,
    ["feature_name", "category", "data_type", "description"]
)

feature_dict_df.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.feature_dictionary")

print(f"Created feature dictionary with {len(feature_dict_data)} features")


# COMMAND ----------


print("GOLD LAYER FEATURE ENGINEERING COMPLETE!")


print(f"Successfully created:")
print(f"   {final_count:,} feature-engineered records")
print(f"   58 total features across 8 categories")
print(f"   Feature correlation analysis")
print(f"   Feature dictionary for documentation")

print(f"Table optimized and analyzed")
print(f" Ready for ML model training!")


# COMMAND ----------


# Display sample features
print("Sample Feature Data (first 5 rows, key features):")
sample_cols = [
    "Date", "Store", "Sales",
    "sales_lag_7", "rolling_mean_30", "demand_volatility",
    "sales_momentum", "competition_pressure", "promo_frequency_30d",
    "is_weekend", "is_holiday"
]
display(spark.table(GOLD_FEATURES).select(sample_cols).limit(5))


# COMMAND ----------


# Feature statistics
print("Feature Statistics:")
display(spark.table(GOLD_FEATURES).select(
    "Sales", "sales_lag_30", "rolling_mean_30", "demand_volatility",
    "sales_momentum", "sales_per_customer"
).describe())

print(" Proceeding to ML model training on Day 4!")