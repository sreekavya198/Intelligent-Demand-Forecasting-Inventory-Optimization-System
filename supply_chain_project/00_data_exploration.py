# Databricks notebook source
import pyspark.sql.connect.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

print("Loading data...")
train_df = spark.read.csv(
    "/Volumes/supply_chain_catalog/bronze_schema/rossmann/train.csv",
    header=True,
    inferSchema=True
)

store_df = spark.read.csv(
    "/Volumes/supply_chain_catalog/bronze_schema/rossmann/store.csv",
    header=True,
    inferSchema=True
)

print(f"Loaded {train_df.count():,} sales records")
print(f"Loaded {store_df.count():,} store records")


# COMMAND ----------

display(train_df.limit(10))

# COMMAND ----------

display(store_df.limit(10))

# COMMAND ----------

train_df.printSchema()

# COMMAND ----------

store_df.printSchema()

# COMMAND ----------

stats = {
    'Total Sales Records': train_df.count(),
    'Total Stores': store_df.count(),
    'Unique Stores in Sales': train_df.select('Store').distinct().count(),
    'Date Range Start': train_df.agg(min('Date')).collect()[0][0],
    'Date Range End': train_df.agg(max('Date')).collect()[0][0],
    'Total Days': train_df.select('Date').distinct().count(),
    # 'Average Daily Sales': round(train_df.agg(F.avg(F.col('Sales'))).collect()[0][0], 2),
    'Max Daily Sales': train_df.agg(max('Sales')).collect()[0][0],
    # 'Total Revenue': round(train_df.agg(sum('Sales')).collect()[0][0], 2)
}

for key, value in stats.items():
    print(f"  {key}: {value:,}")

# COMMAND ----------

for key, value in stats.items():
    print(value)

# COMMAND ----------

missing_df = train_df.select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) 
    for c in train_df.columns
])
display(missing_df)

# COMMAND ----------

sales_stats = train_df.select('Sales').describe()
display(sales_stats)

# COMMAND ----------

store_status = train_df.groupBy('Open').count().orderBy('Open')
display(store_status)

# COMMAND ----------

promo_dist = train_df.groupBy('Promo').count().orderBy('Promo')
display(promo_dist)

# COMMAND ----------

# DBTITLE 1,Sales by day of week
dow_sales = train_df.withColumn('DayOfWeek', col('DayOfWeek').cast('integer')) \
    .groupBy('DayOfWeek') \
    .agg(
        avg('Sales').alias('avg_sales'),
        sum('Sales').alias('total_sales'),
        count('*').alias('num_records')
    ) \
    .orderBy('DayOfWeek')
display(dow_sales)

# COMMAND ----------

# DBTITLE 1,Store type distribution
store_types = store_df.groupBy('StoreType').count().orderBy('count', ascending=False)
display(store_types)

# COMMAND ----------

quality_checks = {
    'Records with Sales=0 when Open=1': train_df.filter((col('Open') == 1) & (col('Sales') == 0)).count(),
    'Records with Sales>0 when Open=0': train_df.filter((col('Open') == 0) & (col('Sales') > 0)).count(),
    'Records with negative Sales': train_df.filter(col('Sales') < 0).count(),
    'Records with null Sales': train_df.filter(col('Sales').isNull()).count(),
    'Duplicate Store-Date combinations': train_df.groupBy('Store', 'Date').count().filter('count > 1').count()
}

for check, count in quality_checks.items():
    status = "PASS" if count == 0 else "ATTENTION"
    print(f"  {status} - {check}: {count:,}")


# COMMAND ----------

summary_dict = {**stats, **quality_checks}


# COMMAND ----------

summary_df = spark.createDataFrame(
    [(k, str(v)) for k, v in summary_dict.items()],
    ['metric', 'value']
)
display(summary_df)


# COMMAND ----------

summary_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("supply_chain_catalog.analytics_schema.data_exploration_summary")

print("Saved exploration summary to analytics_schema.data_exploration_summary")