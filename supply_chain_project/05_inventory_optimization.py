# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from scipy.stats import norm
import numpy as np
import pandas as pd
from pyspark.sql.window import Window
import builtins

# COMMAND ----------


CATALOG = "supply_chain_catalog"
ML_SCHEMA = "ml_schema"
GOLD_SCHEMA = "gold_schema"

ML_PREDICTIONS = f"{CATALOG}.{ML_SCHEMA}.demand_predictions"
GOLD_FEATURES = f"{CATALOG}.{GOLD_SCHEMA}.demand_features"
INVENTORY_RECOMMENDATIONS = f"{CATALOG}.{ML_SCHEMA}.inventory_recommendations"
BUSINESS_IMPACT = f"{CATALOG}.{ML_SCHEMA}.business_impact_analysis"


# COMMAND ----------

# Business parameters
SERVICE_LEVEL = 0.95  # 95% service level (avoid 5% stockouts)
LEAD_TIME_DAYS = 7    # Supplier lead time
HOLDING_COST_PER_UNIT_PER_MONTH = 2.50  # $2.50/unit/month
STOCKOUT_COST_PER_UNIT = 50.0  # Lost profit per stockout
CURRENT_INVENTORY_POLICY_DAYS = 45  # Current: 45 days of inventory
ORDER_COST = 100  # Fixed cost per order


# COMMAND ----------


class InventoryOptimizer:
    """
    Advanced inventory optimization using statistical methods
    and machine learning predictions
    """
    
    def __init__(self, service_level=0.95, lead_time_days=7):
        self.service_level = service_level
        self.lead_time_days = lead_time_days
        self.z_score = norm.ppf(service_level)
    
    def calculate_safety_stock(self, demand_std, lead_time_days=None):
        """
        Calculate safety stock using statistical formula
        
        Safety Stock = Z-score × σ_demand × √lead_time
        
        Args:
            demand_std: Standard deviation of demand
            lead_time_days: Supplier lead time in days
        
        Returns:
            float: Safety stock quantity
        """
        if lead_time_days is None:
            lead_time_days = self.lead_time_days
        
        safety_stock = self.z_score * demand_std * np.sqrt(lead_time_days)
        return builtins.max(0, safety_stock)  # Cannot be negative
    
    def calculate_reorder_point(self, avg_daily_demand, demand_std, lead_time_days=None):
        """
        Calculate reorder point
        
        Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock
        
        Args:
            avg_daily_demand: Average daily demand
            demand_std: Standard deviation of daily demand
            lead_time_days: Supplier lead time in days
        
        Returns:
            float: Reorder point quantity
        """
        if lead_time_days is None:
            lead_time_days = self.lead_time_days
        
        safety_stock = self.calculate_safety_stock(demand_std, lead_time_days)
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        return builtins.max(0, reorder_point)
    
    def calculate_economic_order_quantity(self, annual_demand, order_cost, holding_cost):
        """
        Calculate Economic Order Quantity (EOQ)
        
        EOQ = √(2 × Annual Demand × Order Cost / Holding Cost per Unit)
        
        Args:
            annual_demand: Annual demand quantity
            order_cost: Fixed cost per order
            holding_cost: Annual holding cost per unit
        
        Returns:
            float: Optimal order quantity
        """
        if holding_cost <= 0 or annual_demand <= 0:
            return annual_demand / 12  # Default to 1 month supply
        
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost)
        return builtins.max(1, eoq)
    
    def calculate_optimal_inventory(
        self, 
        forecast_30d, 
        demand_volatility, 
        current_inventory=None,
        lead_time_days=None
    ):
        """
        Calculate comprehensive inventory optimization metrics
        
        Args:
            forecast_30d: 30-day demand forecast
            demand_volatility: Coefficient of variation (std/mean)
            current_inventory: Current inventory level
            lead_time_days: Supplier lead time
        
        Returns:
            dict: Complete inventory recommendations
        """
        if lead_time_days is None:
            lead_time_days = self.lead_time_days
        
        # Calculate metrics
        avg_daily_demand = forecast_30d / 30
        demand_std = avg_daily_demand * demand_volatility
        
        # Safety stock
        safety_stock = self.calculate_safety_stock(demand_std, lead_time_days)
        
        # Reorder point
        reorder_point = self.calculate_reorder_point(
            avg_daily_demand, demand_std, lead_time_days
        )
        
        # Optimal order quantity (simplified - using 30-day supply)
        optimal_order_qty = forecast_30d
        
        # Economic order quantity
        annual_demand = forecast_30d * 12
        holding_cost_annual = HOLDING_COST_PER_UNIT_PER_MONTH * 12
        eoq = self.calculate_economic_order_quantity(
            annual_demand, ORDER_COST, holding_cost_annual
        )
        
        # Stockout risk assessment
        if current_inventory is not None:
            stockout_risk = "HIGH" if current_inventory < reorder_point else "LOW"
            days_until_stockout = (current_inventory / avg_daily_demand) if avg_daily_demand > 0 else 999
        else:
            stockout_risk = "UNKNOWN"
            days_until_stockout = None
        
        return {
            'avg_daily_demand': avg_daily_demand,
            'demand_std': demand_std,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'optimal_order_qty': optimal_order_qty,
            'economic_order_qty': eoq,
            'stockout_risk': stockout_risk,
            'days_until_stockout': days_until_stockout,
            'service_level': self.service_level * 100
        }


# COMMAND ----------


# Load predictions
predictions_df = spark.table(ML_PREDICTIONS)
print(f"Loaded {predictions_df.count():,} predictions")

# Load features for volatility data
features_df = spark.table(GOLD_FEATURES)
print(f"Loaded {features_df.count():,} feature records")

# Join predictions with volatility features
enriched_df = predictions_df.alias("pred").join(
    features_df.select("Store", "Date", "demand_volatility", "rolling_std_30").alias("feat"),
    (col("pred.Store") == col("feat.Store")) & (col("pred.Date") == col("feat.Date")),
    "left"
).drop(col('feat.Store')).drop(col('feat.Date'))

print(f"Enriched predictions with volatility metrics")


# COMMAND ----------


# Window for 30-day forecast
window_30d_future = Window.partitionBy("Store").orderBy("Date").rowsBetween(0, 29)

forecast_df = enriched_df \
    .withColumn("forecast_30d", 
        sum("predicted_sales_xgboost").over(window_30d_future)
    ) \
    .withColumn("avg_volatility_30d",
        avg("demand_volatility").over(window_30d_future)
    )


# COMMAND ----------


# Get latest forecast for each store
latest_forecast = forecast_df \
    .groupBy("Store") \
    .agg(
        max("Date").alias("latest_date"),
        last("forecast_30d").alias("forecast_next_30d"),
        last("avg_volatility_30d").alias("demand_volatility_30d"),
        avg("actual_sales").alias("avg_actual_sales")
    )

print(f"Generated forecasts for {latest_forecast.count():,} stores")


# COMMAND ----------


print(" Applying inventory optimization algorithms...")

# Initialize optimizer
optimizer = InventoryOptimizer(service_level=SERVICE_LEVEL, lead_time_days=LEAD_TIME_DAYS)

# Convert to Pandas for optimization (UDF approach would be slower)
forecast_pdf = latest_forecast.toPandas()

# Apply optimization
optimization_results = []

for idx, row in forecast_pdf.iterrows():
    store = row['Store']
    forecast_30d = row['forecast_next_30d']
    volatility = row['demand_volatility_30d']
    
    # Assume current inventory (in real scenario, would come from inventory system)
    # For demo: random between 0 and 60 days of supply
    current_inventory = np.random.uniform(0, 60) * (forecast_30d / 30)
    
    # Calculate optimization
    opt_result = optimizer.calculate_optimal_inventory(
        forecast_30d=forecast_30d,
        demand_volatility=volatility,
        current_inventory=current_inventory,
        lead_time_days=LEAD_TIME_DAYS
    )
    
    optimization_results.append({
        'Store': store,
        'latest_date': row['latest_date'],
        'forecast_next_30d': forecast_30d,
        'demand_volatility': volatility,
        'current_inventory': current_inventory,
        'avg_daily_demand': opt_result['avg_daily_demand'],
        'demand_std': opt_result['demand_std'],
        'safety_stock': opt_result['safety_stock'],
        'reorder_point': opt_result['reorder_point'],
        'optimal_order_qty': opt_result['optimal_order_qty'],
        'economic_order_qty': opt_result['economic_order_qty'],
        'stockout_risk': opt_result['stockout_risk'],
        'days_until_stockout': opt_result['days_until_stockout'],
        'service_level': opt_result['service_level']
    })
    
    if idx % 100 == 0:
        print(f"  Processed {idx}/{len(forecast_pdf)} stores...")

print(f"Optimization complete for all {len(optimization_results)} stores")


# COMMAND ----------

optimization_df = spark.createDataFrame(pd.DataFrame(optimization_results))


# COMMAND ----------


print(" Generating business recommendations...")

recommendations_df = optimization_df \
    .withColumn("recommended_action",
        when(col("stockout_risk") == "HIGH", "URGENT: Place order immediately")
        .when(col("current_inventory") < col("safety_stock"), "WARNING: Below safety stock")
        .when(col("current_inventory") > col("optimal_order_qty") * 2, "EXCESS: Reduce inventory")
        .otherwise("NORMAL: Monitor")
    ) \
    .withColumn("order_priority",
        when(col("stockout_risk") == "HIGH", 1)
        .when(col("current_inventory") < col("safety_stock"), 2)
        .when(col("current_inventory") < col("reorder_point"), 3)
        .otherwise(4)
    ) \
    .withColumn("inventory_gap",
        col("reorder_point") - col("current_inventory")
    ) \
    .withColumn("excess_inventory",
        when(col("current_inventory") > col("optimal_order_qty") * 2,
            col("current_inventory") - col("optimal_order_qty")
        ).otherwise(0)
    ) \
    .withColumn("recommendation_timestamp", current_timestamp())

print("Business recommendations generated")


# COMMAND ----------


print(" Saving inventory recommendations...")

recommendations_df.write.format("delta").mode("overwrite") \
    .saveAsTable(INVENTORY_RECOMMENDATIONS)

# Add table comment
spark.sql(f"""
    COMMENT ON TABLE {INVENTORY_RECOMMENDATIONS} IS
    'AI-powered inventory optimization recommendations.
    Includes safety stock, reorder points, optimal order quantities,
    stockout risk assessment, and actionable recommendations.
    Uses 95% service level and {LEAD_TIME_DAYS}-day lead time assumptions.'
""")

print(f"Saved recommendations to {INVENTORY_RECOMMENDATIONS}")


# COMMAND ----------


print("BUSINESS IMPACT ANALYSIS")


# Calculate savings and improvements
impact_analysis = recommendations_df.agg(
    count("Store").alias("total_stores"),
    sum("forecast_next_30d").alias("total_forecasted_demand"),
    sum("current_inventory").alias("total_current_inventory"),
    sum("optimal_order_qty").alias("total_optimal_inventory"),
    sum("safety_stock").alias("total_safety_stock"),
    sum(when(col("stockout_risk") == "HIGH", 1).otherwise(0)).alias("high_risk_stores"),
    sum("excess_inventory").alias("total_excess_inventory")
).collect()[0]

# Current inventory value
current_inventory_value = impact_analysis['total_current_inventory'] * HOLDING_COST_PER_UNIT_PER_MONTH

# Optimal inventory value
optimal_inventory_value = impact_analysis['total_optimal_inventory'] * HOLDING_COST_PER_UNIT_PER_MONTH

# Calculate savings
inventory_reduction = impact_analysis['total_current_inventory'] - impact_analysis['total_optimal_inventory']
monthly_savings = inventory_reduction * HOLDING_COST_PER_UNIT_PER_MONTH
annual_savings = monthly_savings * 12

# Stockout prevention
current_stockout_rate = 0.15  # Assume 15% current stockout rate (industry avg)
target_stockout_rate = 1 - SERVICE_LEVEL  # 5% with optimization
stockouts_prevented = (current_stockout_rate - target_stockout_rate) * impact_analysis['total_forecasted_demand']
stockout_cost_savings = stockouts_prevented * STOCKOUT_COST_PER_UNIT

# Total ROI
total_annual_savings = annual_savings + stockout_cost_savings


# COMMAND ----------


print(f" BUSINESS IMPACT SUMMARY:")

print(f"Total Stores Analyzed: {impact_analysis['total_stores']:,}")
print(f"Forecasted 30-Day Demand: {impact_analysis['total_forecasted_demand']:,.0f} units")
print(f"\nINVENTORY OPTIMIZATION:")
print(f"  Current Inventory: {impact_analysis['total_current_inventory']:,.0f} units")
print(f"  Optimal Inventory: {impact_analysis['total_optimal_inventory']:,.0f} units")
print(f"  Inventory Reduction: {inventory_reduction:,.0f} units ({inventory_reduction/impact_analysis['total_current_inventory']*100:.1f}%)")
print(f"  Excess Inventory: {impact_analysis['total_excess_inventory']:,.0f} units")
print(f"\nCOST SAVINGS:")
print(f"  Current Monthly Holding Cost: ${current_inventory_value:,.2f}")
print(f"  Optimized Monthly Holding Cost: ${optimal_inventory_value:,.2f}")
print(f"  Monthly Savings: ${monthly_savings:,.2f}")
print(f"  Annual Holding Cost Savings: ${annual_savings:,.2f}")
print(f"\nSTOCKOUT PREVENTION:")
print(f"  High-Risk Stores: {impact_analysis['high_risk_stores']:,}")
print(f"  Current Stockout Rate: {current_stockout_rate*100:.1f}%")
print(f"  Target Stockout Rate: {target_stockout_rate*100:.1f}%")
print(f"  Stockouts Prevented: {stockouts_prevented:,.0f} incidents")
print(f"  Stockout Cost Savings: ${stockout_cost_savings:,.2f}")
print(f"\nTOTAL ANNUAL ROI: ${total_annual_savings:,.2f}")


# COMMAND ----------

from datetime import datetime
# Save impact analysis
impact_df = spark.createDataFrame([{
    'analysis_date': datetime.utcnow(),
    'total_stores': int(impact_analysis['total_stores']),
    'forecasted_demand_30d': float(impact_analysis['total_forecasted_demand']),
    'current_inventory': float(impact_analysis['total_current_inventory']),
    'optimal_inventory': float(impact_analysis['total_optimal_inventory']),
    'inventory_reduction_units': float(inventory_reduction),
    'inventory_reduction_pct': float(inventory_reduction/impact_analysis['total_current_inventory']*100),
    'monthly_holding_cost_savings': float(monthly_savings),
    'annual_holding_cost_savings': float(annual_savings),
    'high_risk_stores': int(impact_analysis['high_risk_stores']),
    'stockouts_prevented': float(stockouts_prevented),
    'stockout_cost_savings': float(stockout_cost_savings),
    'total_annual_roi': float(total_annual_savings),
    'service_level_target': SERVICE_LEVEL * 100,
    'lead_time_days': LEAD_TIME_DAYS
}])

impact_df.write.format("delta").mode("append") \
    .saveAsTable(BUSINESS_IMPACT)

spark.sql(f"""
    COMMENT ON TABLE {BUSINESS_IMPACT} IS
    'Business impact analysis and ROI calculations for inventory optimization.
    Tracks cost savings, stockout prevention, and total ROI over time.'
""")

print(f"\nBusiness impact analysis saved to {BUSINESS_IMPACT}")


# COMMAND ----------



print("TOP PRIORITY ACTIONS")


# Get top 20 urgent actions
urgent_actions = spark.table(INVENTORY_RECOMMENDATIONS) \
    .filter(col("stockout_risk") == "HIGH") \
    .orderBy(col("days_until_stockout").asc()) \
    .select(
        "Store",
        "current_inventory",
        "reorder_point",
        "optimal_order_qty",
        "days_until_stockout",
        "recommended_action"
    ) \
    .limit(20)

print("  TOP 20 URGENT REORDER RECOMMENDATIONS:")
display(urgent_actions)


# COMMAND ----------


print("INVENTORY OPTIMIZATION COMPLETE!")


print(f" Optimization Results:")
print(f" {impact_analysis['total_stores']:,} stores optimized")
print(f" {impact_analysis['high_risk_stores']:,} high-risk stores identified")
print(f" ${total_annual_savings:,.2f} annual ROI projected")
print(f" {(inventory_reduction/impact_analysis['total_current_inventory']*100):.1f}% inventory reduction")
print(f" {((current_stockout_rate - target_stockout_rate)/current_stockout_rate*100):.1f}% stockout reduction")

print(f" Deliverables created:")
print(f" Inventory recommendations table")
print(f" Business impact analysis")
print(f" Priority action list")