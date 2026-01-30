# Databricks notebook source

from databricks.sdk import WorkspaceClient
import json


# COMMAND ----------


workflow_definition = {
    "name": "Supply_Chain_ML_Pipeline_Daily",
    "email_notifications": {
        "on_failure": ["sreekavya198@gmail.com"]
    },
    "timeout_seconds": 14400,  # 4 hours
    "max_concurrent_runs": 1,
    "tasks": [
        {
            "task_key": "bronze_ingestion",
            "description": "Ingest raw data into Bronze layer",
            "notebook_task": {
                "notebook_path": "/Users/sreekavya198@gmail.com/supply_chain_project/01_bronze_ingestion",
                "base_parameters": {}
            },
            "new_cluster": {
                "spark_version": "13.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2,
                "spark_conf": {
                    "spark.databricks.delta.preview.enabled": "true"
                }
            },
            "timeout_seconds": 3600,
            "max_retries": 2
        },
        {
            "task_key": "silver_cleaning",
            "description": "Clean and validate data in Silver layer",
            "depends_on": [{"task_key": "bronze_ingestion"}],
            "notebook_task": {
                "notebook_path": "/Users/sreekavya198@gmail.com/supply_chain_project/02_silver_cleaning",
                "base_parameters": {}
            },
            "timeout_seconds": 3600,
            "max_retries": 2
        },
        {
            "task_key": "gold_features",
            "description": "Engineer features in Gold layer",
            "depends_on": [{"task_key": "silver_cleaning"}],
            "notebook_task": {
                "notebook_path": "/Users/sreekavya198@gmail.com/supply_chain_project/03_gold_features",
                "base_parameters": {}
            },
            "timeout_seconds": 3600,
            "max_retries": 1
        },
        {
            "task_key": "ml_predictions",
            "description": "Generate ML predictions",
            "depends_on": [{"task_key": "gold_features"}],
            "notebook_task": {
                "notebook_path": "/Users/sreekavya198@gmail.com/supply_chain_project/04_ml_model_training",
                "base_parameters": {}
            },
            "timeout_seconds": 3600,
            "max_retries": 1
        },
        {
            "task_key": "inventory_optimization",
            "description": "Calculate inventory recommendations",
            "depends_on": [{"task_key": "ml_predictions"}],
            "notebook_task": {
                "notebook_path": "/Users/sreekavya198@gmail.com/supply_chain_project/05_inventory_optimization",
                "base_parameters": {}
            },
            "timeout_seconds": 1800,
            "max_retries": 1
        }
    ],
    "schedule": {
        "quartz_cron_expression": "0 0 2 * * ?",  # Daily at 2 AM
        "timezone_id": "UTC",
        "pause_status": "UNPAUSED"
    },
    "format": "MULTI_TASK"
}

# COMMAND ----------

with open("/Volumes/supply_chain_catalog/analytics_schema/workflow_json/workflow_definition.json", "w") as f:
    json.dump(workflow_definition, f, indent=2)