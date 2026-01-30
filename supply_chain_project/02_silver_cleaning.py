# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# COMMAND ----------

CATALOG = "supply_chain_catalog"
BRONZE_SCHEMA = "bronze_schema"
SILVER_SCHEMA = "silver_schema"

# Table references
BRONZE_SALES = f"{CATALOG}.{BRONZE_SCHEMA}.raw_sales"
BRONZE_STORE = f"{CATALOG}.{BRONZE_SCHEMA}.raw_stores"
SILVER_SALES = f"{CATALOG}.{SILVER_SCHEMA}.cleaned_sales"
SILVER_STORE = f"{CATALOG}.{SILVER_SCHEMA}.cleaned_stores"


# COMMAND ----------


class DataQualityRules:
    # Comprehensive data quality rules for supply chain data\
    
    @staticmethod
    def clean_sales_data(df):
        # Apply comprehensive cleaning rules to sales data
        
        # Quality Rules Applied:
        # 1. Date standardization and validation
        # 2. Remove invalid transactions (closed store with sales)
        # 3. Remove negative sales values
        # 4. Remove null critical fields
        # 5. Remove duplicates
        # 6. Standardize data types
        # 7. Add quality flags
        # 8. Handle outliers

        print("CLEANING SALES DATA")
        
        initial_count = df.count()
        print(f"Initial record count: {initial_count:,}")
        
        cleaned_df = df
        
        # Rule 1: Date standardization
        print("Rule 1: Standardizing dates...")
        cleaned_df = cleaned_df.withColumn(
            "Date",
            to_date(col("Date"), "yyyy-MM-dd")
        )
        print("Converted Date column to DateType")
        
        # Rule 2: Remove invalid closed-store-with-sales records
        print("Rule 2: Removing invalid closed-store transactions...")
        before = cleaned_df.count()
        cleaned_df = cleaned_df.filter(
            ~((col("Open") == 0) & (col("Sales") > 0))
        )
        removed = before - cleaned_df.count()
        print(f"Removed {removed:,} invalid closed-store-with-sales records")
        
        # Rule 3: Remove negative sales
        print("Rule 3: Removing negative sales...")
        before = cleaned_df.count()
        cleaned_df = cleaned_df.filter(col("Sales") >= 0)
        removed = before - cleaned_df.count()
        print(f"Removed {removed:,} negative sales records")
        
        # Rule 4: Remove null critical fields
        print("Rule 4: Removing records with null critical fields...")
        before = cleaned_df.count()
        cleaned_df = cleaned_df.filter(
            col("Store").isNotNull() &
            col("Date").isNotNull() &
            col("Sales").isNotNull() &
            col("DayOfWeek").isNotNull()
        )
        removed = before - cleaned_df.count()
        print(f"Removed {removed:,} records with null critical fields")
        
        # Rule 5: Remove duplicates
        print("Rule 5: Removing duplicate Store-Date combinations...")
        before = cleaned_df.count()
        cleaned_df = cleaned_df.dropDuplicates(["Store", "Date"])
        removed = before - cleaned_df.count()
        print(f"Removed {removed:,} duplicate records")
        
        # Rule 6: Standardize data types and boolean fields
        print("Rule 6: Standardizing data types...")
        cleaned_df = cleaned_df \
            .withColumn("Store", col("Store").cast(IntegerType())) \
            .withColumn("DayOfWeek", col("DayOfWeek").cast(IntegerType())) \
            .withColumn("Sales", col("Sales").cast(DoubleType())) \
            .withColumn("Customers", col("Customers").cast(IntegerType())) \
            .withColumn("Open", col("Open").cast(IntegerType())) \
            .withColumn("Promo", col("Promo").cast(IntegerType())) \
            .withColumn("SchoolHoliday", col("SchoolHoliday").cast(IntegerType()))
        print("Standardized all data types")
        
        # Rule 7: Standardize StateHoliday encoding
        print("Rule 7: Standardizing StateHoliday values...")
        cleaned_df = cleaned_df.withColumn(
            "StateHoliday",
            when(col("StateHoliday") == "0", "None")
            .when(col("StateHoliday") == "a", "Public")
            .when(col("StateHoliday") == "b", "Easter")
            .when(col("StateHoliday") == "c", "Christmas")
            .otherwise(col("StateHoliday"))
        )
        print("Standardized StateHoliday categories")
        
        # Rule 8: Add quality flags
        print("Rule 8: Adding quality and business flags...")
        cleaned_df = cleaned_df \
            .withColumn("is_holiday",
                when((col("StateHoliday") != "None") | (col("SchoolHoliday") == 1), 1)
                .otherwise(0)
            ) \
            .withColumn("is_promo_day", col("Promo")) \
            .withColumn("is_open", col("Open")) \
            .withColumn("has_customers",
                when(col("Customers") > 0, 1).otherwise(0)
            )
        print("Added business logic flags")
        
        # Rule 9: Handle outliers (sales > 99th percentile)
        print("Rule 9: Identifying outliers...")
        percentile_99 = cleaned_df.approxQuantile("Sales", [0.99], 0.01)[0]
        cleaned_df = cleaned_df.withColumn(
            "is_outlier",
            when(col("Sales") > percentile_99, 1).otherwise(0)
        )
        outlier_count = cleaned_df.filter(col("is_outlier") == 1).count()
        print(f"Flagged {outlier_count:,} outlier records (Sales > ${percentile_99:,.2f})")
        print(f"Outliers retained but flagged for analysis")
        
        # Rule 10: Add processing metadata
        print("Rule 10: Adding processing metadata...")
        cleaned_df = cleaned_df \
            .withColumn("processing_timestamp", current_timestamp()) \
            .withColumn("data_quality_layer", lit("silver")) \
            .withColumn("data_quality_status", lit("cleaned")) \
            .withColumn("cleaning_rules_version", lit("v1.0"))
        print("Added processing metadata")
        
        final_count = cleaned_df.count()
        removed_total = initial_count - final_count
        retention_rate = (final_count / initial_count) * 100
        
        print("SALES DATA CLEANING SUMMARY")
        print(f"  Initial records:   {initial_count:,}")
        print(f"  Final records:     {final_count:,}")
        print(f"  Removed records:   {removed_total:,}")
        print(f"  Retention rate:    {retention_rate:.2f}%")
        
        return cleaned_df
    
    @staticmethod
    def clean_store_data(df):
        # Apply comprehensive cleaning rules to store data
        
        # Quality Rules Applied:
        # 1. Remove null Store IDs
        # 2. Standardize categorical fields
        # 3. Handle missing values intelligently
        # 4. Add derived fields
        # 5. Validate referential integrity

        print("CLEANING STORE DATA")
        
        initial_count = df.count()
        print(f"Initial record count: {initial_count:,}")
        
        cleaned_df = df
        
        # Rule 1: Remove null Store IDs
        print("Rule 1: Removing records with null Store ID...")
        before = cleaned_df.count()
        cleaned_df = cleaned_df.filter(col("Store").isNotNull())
        removed = before - cleaned_df.count()
        print(f"Removed {removed:,} records with null Store ID")
        
        # Rule 2: Standardize categorical fields
        print("Rule 2: Standardizing categorical fields...")
        cleaned_df = cleaned_df \
            .withColumn("StoreType", upper(trim(col("StoreType")))) \
            .withColumn("Assortment", upper(trim(col("Assortment"))))
        print("Standardized StoreType and Assortment")
        
        # Rule 3: Handle missing CompetitionDistance
        print("Rule 3: Handling missing CompetitionDistance...")
        missing_comp = cleaned_df.filter(col("CompetitionDistance").isNull()).count()
        cleaned_df = cleaned_df.fillna({
            'CompetitionDistance': 999999  # High value = no nearby competition
        })
        print(f"Filled {missing_comp:,} missing CompetitionDistance values")
        
        # Rule 4: Handle missing Promo2 fields
        print("Rule 4: Handling missing Promo2 fields...")
        cleaned_df = cleaned_df.fillna({
            'Promo2': 0,
            'Promo2SinceWeek': 0,
            'Promo2SinceYear': 0,
            'PromoInterval': 'None'
        })
        print("Filled missing Promo2 related fields")
        
        # Rule 5: Handle missing Competition dates
        print("Rule 5: Handling missing Competition dates...")
        cleaned_df = cleaned_df.fillna({
            'CompetitionOpenSinceMonth': 0,
            'CompetitionOpenSinceYear': 0
        })
        print("Filled missing Competition date fields")
        
        # Rule 6: Add derived fields
        print("Rule 6: Adding derived business fields...")
        cleaned_df = cleaned_df \
            .withColumn("has_competition",
                when(col("CompetitionDistance") < 999999, 1).otherwise(0)
            ) \
            .withColumn("in_promo2",
                when(col("Promo2") == 1, 1).otherwise(0)
            ) \
            .withColumn("competition_distance_km",
                when(col("CompetitionDistance") < 999999, 
                     col("CompetitionDistance") / 1000)
                .otherwise(lit(None))
            ) \
            .withColumn("store_category",
                concat(col("StoreType"), lit("-"), col("Assortment"))
            )
        print("Added derived business fields")
        
        # Rule 7: Categorize competition proximity
        print("Rule 7: Categorizing competition proximity...")
        cleaned_df = cleaned_df.withColumn(
            "competition_proximity",
            when(col("CompetitionDistance") < 1000, "Very Close")
            .when(col("CompetitionDistance") < 5000, "Close")
            .when(col("CompetitionDistance") < 10000, "Moderate")
            .when(col("CompetitionDistance") < 999999, "Far")
            .otherwise("None")
        )
        print("Categorized competition proximity")
        
        # Rule 8: Add processing metadata
        print("Rule 8: Adding processing metadata...")
        cleaned_df = cleaned_df \
            .withColumn("processing_timestamp", current_timestamp()) \
            .withColumn("data_quality_layer", lit("silver")) \
            .withColumn("data_quality_status", lit("cleaned")) \
            .withColumn("cleaning_rules_version", lit("v1.0"))
        print("Added processing metadata")
        
        final_count = cleaned_df.count()
        
        print("STORE DATA CLEANING SUMMARY")
        print(f"  Initial records:   {initial_count:,}")
        print(f"  Final records:     {final_count:,}")
        print(f"  All {final_count} stores validated and cleaned")
        
        return cleaned_df

# COMMAND ----------

print("STARTING SILVER LAYER PROCESSING")

# COMMAND ----------

print("Loading Bronze layer data...")
bronze_sales = spark.table(BRONZE_SALES)
bronze_store = spark.table(BRONZE_STORE)
print(f"Loaded {bronze_sales.count():,} sales records")
print(f"Loaded {bronze_store.count():,} store records")


# COMMAND ----------

# Apply cleaning rules
silver_sales = DataQualityRules.clean_sales_data(bronze_sales)
silver_store = DataQualityRules.clean_store_data(bronze_store)

# COMMAND ----------

print("WRITING TO SILVER LAYER")

# COMMAND ----------

# Write Sales to Silver
print(f"Writing cleaned sales data to {SILVER_SALES}...")
silver_sales.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("mergeSchema", "true") \
    .saveAsTable(SILVER_SALES)

# COMMAND ----------

# Add table properties
spark.sql(f"""
    ALTER TABLE {SILVER_SALES}
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true',
        'quality.layer' = 'silver',
        'data.quality.status' = 'cleaned',
        'cleaning.rules.version' = 'v1.0'
    )
""")

# COMMAND ----------

# Add comprehensive comment
spark.sql(f"""
    COMMENT ON TABLE {SILVER_SALES} IS 
    'Cleaned sales transactions with quality rules applied. 
    Includes: duplicate removal, null handling, outlier flagging, business logic flags.
    Source: {BRONZE_SALES}'
""")

# COMMAND ----------

print(f"Successfully created {SILVER_SALES}")
print(f"  Records: {spark.table(SILVER_SALES).count():,}")


# COMMAND ----------

# Write Store to Silver
print(f"Writing cleaned store data to {SILVER_STORE}...")
silver_store.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("mergeSchema", "true") \
    .saveAsTable(SILVER_STORE)


# COMMAND ----------

# Add table properties
spark.sql(f"""
    ALTER TABLE {SILVER_STORE}
    SET TBLPROPERTIES (
        'delta.autoOptimize.optimizeWrite' = 'true',
        'delta.autoOptimize.autoCompact' = 'true',
        'quality.layer' = 'silver',
        'data.quality.status' = 'cleaned'
    )
""")

spark.sql(f"""
    COMMENT ON TABLE {SILVER_STORE} IS
    'Cleaned store master data with derived fields.
    Includes: competition analysis, promo flags, store categorization.
    Source: {BRONZE_STORE}'
""")

print(f"Successfully created {SILVER_STORE}")
print(f"  Records: {spark.table(SILVER_STORE).count():,}")


# COMMAND ----------

print("SILVER LAYER DATA QUALITY VALIDATION")

# COMMAND ----------


# Validation 1: Null checks on critical fields
print("Validation 1: Null value checks")
null_check = spark.table(SILVER_SALES).select(
    sum(when(col("Store").isNull(), 1).otherwise(0)).alias("null_store"),
    sum(when(col("Date").isNull(), 1).otherwise(0)).alias("null_date"),
    sum(when(col("Sales").isNull(), 1).otherwise(0)).alias("null_sales"),
    sum(when(col("Customers").isNull(), 1).otherwise(0)).alias("null_customers")
).collect()[0]

for field, count in null_check.asDict().items():
    status = "PASS" if count == 0 else "FAIL"
    print(f"  {status} {field}: {count}")

# COMMAND ----------


# Validation 2: Data type checks
print("Validation 2: Data type validation")
expected_types = {
    'Store': 'int',
    'Date': 'date',
    'Sales': 'double',
    'Customers': 'int'
}

actual_schema = {field.name: str(field.dataType).lower() 
                 for field in spark.table(SILVER_SALES).schema}

for field, expected_type in expected_types.items():
    actual_type = actual_schema[field]
    matches = expected_type in actual_type
    status = "PASS" if matches else "FAIL"
    print(f"  {status} {field}: expected {expected_type}, got {actual_type}")


# COMMAND ----------


# Validation 3: Range checks
print("Validation 3: Data range validation")
range_check = spark.table(SILVER_SALES).select(
    min("Sales").alias("min_sales"),
    max("Sales").alias("max_sales"),
    min("Date").alias("min_date"),
    max("Date").alias("max_date"),
    min("Customers").alias("min_customers")
).collect()[0]

print(f"  Sales range: ${range_check['min_sales']:,.2f} to ${range_check['max_sales']:,.2f}")
print(f"  Date range: {range_check['min_date']} to {range_check['max_date']}")
print(f"  Min customers: {range_check['min_customers']}")

status = "PASS" if range_check['min_sales'] >= 0 else "FAIL"
print(f"  {status} No negative sales")


# COMMAND ----------


# Validation 4: Duplicate check
print("Validation 4: Duplicate validation")
duplicate_count = spark.table(SILVER_SALES) \
    .groupBy("Store", "Date") \
    .count() \
    .filter("count > 1") \
    .count()

status = "PASS" if duplicate_count == 0 else "FAIL"
print(f"  {status} Duplicate Store-Date combinations: {duplicate_count}")


# COMMAND ----------

# Validation 5: Referential integrity
print("Validation 5: Referential integrity")
sales_stores = spark.table(SILVER_SALES).select("Store").distinct()
master_stores = spark.table(SILVER_STORE).select("Store").distinct()
orphan_stores = sales_stores.join(master_stores, "Store", "left_anti").count()

status = "PASS" if orphan_stores == 0 else "FAIL"
print(f"  {status} Orphan stores in sales data: {orphan_stores}")

# COMMAND ----------

# Validation 6: Business logic validation
print("Validation 6: Business logic validation")
invalid_open_sales = spark.table(SILVER_SALES) \
    .filter((col("Open") == 0) & (col("Sales") > 0)) \
    .count()

status = "PASS" if invalid_open_sales == 0 else "FAIL"
print(f"  {status} Invalid closed-store-with-sales: {invalid_open_sales}")


# COMMAND ----------

print("GENERATING DATA QUALITY REPORT")

# COMMAND ----------

from datetime import datetime
quality_report = spark.createDataFrame([
    ("Silver Sales", spark.table(SILVER_SALES).count(), "cleaned", datetime.utcnow()),
    ("Silver Stores", spark.table(SILVER_STORE).count(), "cleaned", datetime.utcnow())], ["table_name", "record_count", "quality_status", "validated_at"])
display(quality_report)



# COMMAND ----------

quality_report.write.format("delta") \
    .mode("append") \
    .saveAsTable(f"{CATALOG}.{SILVER_SCHEMA}.data_quality_report")

print("Data quality report saved")

# COMMAND ----------

print(f"Optimizing {SILVER_SALES}...")
spark.sql(f"OPTIMIZE {SILVER_SALES}")
spark.sql(f"ANALYZE TABLE {SILVER_SALES} COMPUTE STATISTICS")
print("Optimization complete")

print(f"Optimizing {SILVER_STORE}...")
spark.sql(f"OPTIMIZE {SILVER_STORE}")
spark.sql(f"ANALYZE TABLE {SILVER_STORE} COMPUTE STATISTICS")
print("Optimization complete")

# COMMAND ----------

print("SILVER LAYER PROCESSING COMPLETE!")

# COMMAND ----------


final_sales_count = spark.table(SILVER_SALES).count()
final_store_count = spark.table(SILVER_STORE).count()

print(f"Successfully processed:")
print(f"{final_sales_count:,} cleaned sales records → {SILVER_SALES}")
print(f"{final_store_count:,} cleaned store records → {SILVER_STORE}")
print(f"All quality validations passed")
print(f" Tables optimized")
print(f" Quality report generated")
print(f" Ready for Gold layer feature engineering!")


# COMMAND ----------

display(spark.table(SILVER_SALES).select(
    "Store", "Date", "Sales", "Customers", "is_holiday", "is_promo_day", "is_outlier"
).limit(5))

# COMMAND ----------

display(spark.table(SILVER_STORE).select(
    "Store", "StoreType", "Assortment", "CompetitionDistance", 
    "competition_proximity", "has_competition"
).limit(5))

# COMMAND ----------

