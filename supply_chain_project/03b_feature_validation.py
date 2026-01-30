# Databricks notebook source
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# COMMAND ----------


CATALOG = "supply_chain_catalog"
GOLD_SCHEMA = "gold_schema"
GOLD_FEATURES = f"{CATALOG}.{GOLD_SCHEMA}.demand_features"


# COMMAND ----------

# Load features
features_df = spark.table(GOLD_FEATURES)

# COMMAND ----------


print("Validation 1: Checking for nulls in critical features...")

critical_features = [
    "Sales", "sales_lag_7", "sales_lag_30", "rolling_mean_30",
    "demand_volatility", "sales_momentum"
]

null_counts = features_df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in critical_features
]).collect()[0].asDict()

all_pass = True
for feature, null_count in null_counts.items():
    status = "PASS" if null_count == 0 else "⚠ FAIL"
    print(f"  {status} {feature}: {null_count:,} nulls")
    if null_count > 0:
        all_pass = False

if all_pass:
    print("All critical features have no nulls!")


# COMMAND ----------


print("\nValidation 2: Validating feature ranges...")

range_checks = features_df.select(
    min("demand_volatility").alias("min_volatility"),
    max("demand_volatility").alias("max_volatility"),
    min("sales_momentum").alias("min_momentum"),
    max("sales_momentum").alias("max_momentum"),
    min("competition_pressure").alias("min_comp"),
    max("competition_pressure").alias("max_comp")
).collect()[0]

print(f"  Demand volatility range: {range_checks['min_volatility']:.3f} to {range_checks['max_volatility']:.3f}")
print(f"  Sales momentum range: {range_checks['min_momentum']:.3f} to {range_checks['max_momentum']:.3f}")
print(f"  Competition pressure range: {range_checks['min_comp']} to {range_checks['max_comp']}")


# COMMAND ----------


print("Creating feature distribution visualizations...")

# Convert to Pandas for visualization
sample_pdf = features_df.select(
    "Sales", "rolling_mean_30", "demand_volatility", 
    "sales_momentum", "sales_per_customer"
).sample(0.01).toPandas()  # Sample 1% for visualization

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions', fontsize=16)

features_to_plot = ["Sales", "rolling_mean_30", "demand_volatility", 
                    "sales_momentum", "sales_per_customer"]

for idx, feature in enumerate(features_to_plot):
    row = idx // 3
    col = idx % 3
    axes[row, col].hist(sample_pdf[feature].dropna(), bins=50, edgecolor='black')
    axes[row, col].set_title(f'{feature} Distribution')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')

plt.tight_layout()
display(fig)

print("Feature distributions visualized")


# COMMAND ----------

from datetime import datetime
validation_summary = spark.createDataFrame([
    ("Feature Count", len(features_df.columns), datetime.utcnow()),
    ("Record Count", features_df.count(), datetime.utcnow()),
    ("Null Check Status", "PASS" if all_pass else "FAIL", datetime.utcnow())
], ["check_name", "result", "validated_at"])

validation_summary.write.format("delta").mode("append") \
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.feature_validation_report")

print("Feature validation complete!")
print(" Validation report saved")

# COMMAND ----------

