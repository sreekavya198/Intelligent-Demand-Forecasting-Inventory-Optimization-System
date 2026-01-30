# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
from datetime import datetime

# COMMAND ----------

CATALOG = "supply_chain_catalog"
BRONZE_SCHEMA = "bronze_schema"
RAW_DATA_PATH = "/Volumes/supply_chain_catalog/bronze_schema/rossmann/"


# COMMAND ----------

# Unity Catalog table names
BRONZE_SALES_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.raw_sales"
BRONZE_STORE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.raw_stores"

# COMMAND ----------

INGESTION_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
DATA_SOURCE = "Kaggle - Rossmann Store Sales"

# COMMAND ----------


def ingest_to_bronze(source_path: str, table_name: str, source_description: str, file_format: str = "csv"):
    
    
    # Ingests raw data into Unity Catalog Bronze layer
    
    # Args:
    #     source_path: Full path to source file
    #     table_name: Fully qualified Unity Catalog table name
    #     source_description: Description of data source
    #     file_format: File format (csv, json, parquet, etc.)
    
    # Returns:
    #     DataFrame: Ingested data with metadata
    
    print(f"Ingesting: {source_description}")
    print(f"Source: {source_path}")
    print(f"Target: {table_name}")
    
    # Read raw data
    if file_format == "csv":
        raw_df = spark.read.csv(
            source_path,
            header=True,
            inferSchema=True
        )
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    record_count = raw_df.count()
    print(f"Read {record_count:,} records from source")
    
    # Add comprehensive metadata columns
    bronze_df = raw_df \
        .withColumn("ingestion_timestamp", lit(INGESTION_TIMESTAMP).cast(TimestampType())) \
        .withColumn("source_file_path", lit(source_path)) \
        .withColumn("source_system", lit(DATA_SOURCE)) \
        .withColumn("ingestion_batch_id", lit(datetime.now().strftime("%Y%m%d_%H%M%S"))) \
        .withColumn("record_source", lit(source_description)) \
        .withColumn("is_deleted", lit(False)) \
        .withColumn("row_hash", sha2(concat_ws("||", *raw_df.columns), 256))
    
    print("Added metadata columns:")
    print("  - ingestion_timestamp: Record ingestion time")
    print("  - source_file_path: Source file location")
    print("  - source_system: Data source system")
    print("  - ingestion_batch_id: Batch identifier")
    print("  - record_source: Record description")
    print("  - is_deleted: Soft delete flag")
    print("  - row_hash: Data integrity hash")
    
    # Write to Delta Lake with Unity Catalog
    print(f"Writing to Delta Lake...")
    bronze_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .option("mergeSchema", "true") \
        .saveAsTable(table_name)
    
    # Verify table creation
    final_count = spark.table(table_name).count()
    print(f"Table created successfully: {table_name}")
    print(f"Final record count: {final_count:,}")
    
    # Add table properties and comments
    spark.sql(f"""
        ALTER TABLE {table_name} 
        SET TBLPROPERTIES (
            'delta.autoOptimize.optimizeWrite' = 'true',
            'delta.autoOptimize.autoCompact' = 'true',
            'quality.layer' = 'bronze',
            'project' = 'supply_chain_forecasting',
            'created_by' = 'data_engineering_pipeline',
            'last_ingestion_timestamp' = '{INGESTION_TIMESTAMP}'
        )
    """)
    
    print("Set table properties for optimization")
    
    # Display schema
    print(f"Table Schema:")
    spark.table(table_name).printSchema()
    
    return bronze_df


# COMMAND ----------

print("STARTING BRONZE LAYER INGESTION")
print(f"Target Catalog: {CATALOG}")
print(f"Target Schema: {BRONZE_SCHEMA}")
print(f"Ingestion Time: {INGESTION_TIMESTAMP}")

# COMMAND ----------

print("INGESTING SALES DATA...")
sales_bronze = ingest_to_bronze(
    source_path=f"{RAW_DATA_PATH}train.csv",
    table_name=BRONZE_SALES_TABLE,
    source_description="Historical store sales transactions (2013-2015)"
)

# COMMAND ----------

print("INGESTING STORE DATA...")
store_bronze = ingest_to_bronze(
    source_path=f"{RAW_DATA_PATH}store.csv",
    table_name=BRONZE_STORE_TABLE,
    source_description="Store master data and characteristics"
)

# COMMAND ----------

# DBTITLE 1,BRONZE LAYER VALIDATION
# Check 1: Record counts
print("Record Count Validation:")
sales_count = spark.table(BRONZE_SALES_TABLE).count()
store_count = spark.table(BRONZE_STORE_TABLE).count()
print(f"  Sales records: {sales_count:,}")
print(f"  Store records: {store_count:,}")

# COMMAND ----------

# Check 2: Schema validation
print("Schema Validation:")
sales_cols = set(spark.table(BRONZE_SALES_TABLE).columns)
expected_sales_cols = {'Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 
                       'Promo', 'StateHoliday', 'SchoolHoliday'}
metadata_cols = {'ingestion_timestamp', 'source_file_path', 'source_system', 
                 'ingestion_batch_id', 'record_source', 'is_deleted', 'row_hash'}

if expected_sales_cols.issubset(sales_cols) and metadata_cols.issubset(sales_cols):
    print("All expected columns present in sales table")
else:
    missing = expected_sales_cols - sales_cols
    print(f"Missing columns: {missing}")


# COMMAND ----------

# Check 3: Date range validation
print("Date Range Validation:")
date_range = spark.table(BRONZE_SALES_TABLE).agg(
    min("Date").alias("start_date"),
    max("Date").alias("end_date"),
    countDistinct("Date").alias("unique_dates")
).collect()[0]

print(f"Start Date: {date_range['start_date']}")
print(f"End Date: {date_range['end_date']}")
print(f"Unique Dates: {date_range['unique_dates']:,}")
print(f"Coverage: {(date_range['end_date'] - date_range['start_date']).days} days")


# COMMAND ----------

# Check 4: Duplicate detection
print("Duplicate Detection:")
duplicates = spark.table(BRONZE_SALES_TABLE) \
    .groupBy("Store", "Date") \
    .count() \
    .filter("count > 1") \
    .count()
print(f"  Duplicate Store-Date combinations: {duplicates}")


# COMMAND ----------

# Check 5: Null value analysis
print("Null Value Analysis:")
null_counts = spark.table(BRONZE_SALES_TABLE).select([
    sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
    for c in ['Store', 'Date', 'Sales', 'Customers']
]).collect()[0].asDict()

for col_name, null_count in null_counts.items():
    print(f"{col_name}: {null_count:,} nulls")


# COMMAND ----------


# Check 6: Data quality metrics
print("Data Quality Metrics:")
quality_metrics = spark.table(BRONZE_SALES_TABLE).agg(
    sum(when((col("Open") == 0) & (col("Sales") > 0), 1).otherwise(0)).alias("closed_with_sales"),
    sum(when(col("Sales") < 0, 1).otherwise(0)).alias("negative_sales"),
    avg("Sales").alias("avg_sales"),
    stddev("Sales").alias("stddev_sales")
).collect()[0]

print(f"  Closed stores with sales: {quality_metrics['closed_with_sales']:,}")
print(f"  Negative sales records: {quality_metrics['negative_sales']:,}")
print(f"  Average sales: ${quality_metrics['avg_sales']:,.2f}")
print(f"  Sales std dev: ${quality_metrics['stddev_sales']:,.2f}")


# COMMAND ----------

print("CREATING DATA LINEAGE RECORD")

# COMMAND ----------


lineage_df = spark.createDataFrame([
    (
        "bronze_raw_sales",
        BRONZE_SALES_TABLE,
        "train.csv",
        RAW_DATA_PATH,
        sales_count,
        INGESTION_TIMESTAMP,
        "initial_load",
        "SUCCESS"
    ),
    (
        "bronze_raw_stores",
        BRONZE_STORE_TABLE,
        "store.csv",
        RAW_DATA_PATH,
        store_count,
        INGESTION_TIMESTAMP,
        "initial_load",
        "SUCCESS"
    )
], ["table_name", "full_table_name", "source_file", "source_path", 
    "record_count", "load_timestamp", "load_type", "status"])

lineage_df.write.format("delta") \
    .mode("append") \
    .saveAsTable(f"{CATALOG}.{BRONZE_SCHEMA}.data_lineage")

print("Data lineage recorded")


# COMMAND ----------

# DBTITLE 1,OPTIMIZING DELTA TABLES
print(f"Optimizing {BRONZE_SALES_TABLE}...")
spark.sql(f"OPTIMIZE {BRONZE_SALES_TABLE}")
print("Optimization complete")

print(f"Optimizing {BRONZE_STORE_TABLE}...")
spark.sql(f"OPTIMIZE {BRONZE_STORE_TABLE}")
print("Optimization complete")


# COMMAND ----------

display(spark.table(BRONZE_SALES_TABLE).limit(5))

# COMMAND ----------

display(spark.table(BRONZE_STORE_TABLE).limit(5))

# COMMAND ----------

print(f"{sales_count:,} sales records → {BRONZE_SALES_TABLE}")
print(f"{store_count:,} store records → {BRONZE_STORE_TABLE}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Verify tables exist
# MAGIC SHOW TABLES IN supply_chain_catalog.bronze_schema;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check record counts
# MAGIC SELECT 'Sales' as table_name, COUNT(*) as record_count 
# MAGIC FROM supply_chain_catalog.bronze_schema.raw_sales
# MAGIC UNION ALL
# MAGIC SELECT 'Stores' as table_name, COUNT(*) as record_count 
# MAGIC FROM supply_chain_catalog.bronze_schema.raw_stores;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View sample data
# MAGIC SELECT * FROM supply_chain_catalog.bronze_schema.raw_sales LIMIT 10;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check data quality
# MAGIC SELECT 
# MAGIC     COUNT(*) as total_records,
# MAGIC     COUNT(DISTINCT Store) as unique_stores,
# MAGIC     MIN(Date) as start_date,
# MAGIC     MAX(Date) as end_date,
# MAGIC     AVG(Sales) as avg_sales
# MAGIC FROM supply_chain_catalog.bronze_schema.raw_sales;