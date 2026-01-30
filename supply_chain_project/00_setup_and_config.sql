-- Databricks notebook source
-- Create catalog
CREATE CATALOG IF NOT EXISTS supply_chain_catalog
COMMENT 'Supply chain demand forecasting and inventory optimization project';

-- COMMAND ----------

-- Use the catalog
USE CATALOG supply_chain_catalog;


-- COMMAND ----------


-- Create schemas with proper organization
CREATE SCHEMA IF NOT EXISTS bronze_schema 
COMMENT 'Raw data ingestion layer - immutable source data';

CREATE SCHEMA IF NOT EXISTS silver_schema 
COMMENT 'Cleaned and validated data layer';

CREATE SCHEMA IF NOT EXISTS gold_schema 
COMMENT 'Business-level aggregated and feature-engineered data';

CREATE SCHEMA IF NOT EXISTS ml_schema 
COMMENT 'ML models, predictions, and model artifacts';

CREATE SCHEMA IF NOT EXISTS analytics_schema
COMMENT 'Pre-aggregated tables for dashboards and reporting';


-- COMMAND ----------

-- Verify setup
SHOW SCHEMAS IN supply_chain_catalog;

-- COMMAND ----------

-- MAGIC %py display(dbutils.fs.ls("/Volumes/supply_chain_catalog/bronze_schema/rossmann/"))