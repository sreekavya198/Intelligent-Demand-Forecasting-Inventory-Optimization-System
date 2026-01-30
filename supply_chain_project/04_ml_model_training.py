# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------


CATALOG = "supply_chain_catalog"
GOLD_SCHEMA = "gold_schema"
ML_SCHEMA = "ml_schema"

GOLD_FEATURES = f"{CATALOG}.{GOLD_SCHEMA}.demand_features"
ML_PREDICTIONS = f"{CATALOG}.{ML_SCHEMA}.demand_predictions"


# COMMAND ----------


EXPERIMENT_NAME = "/Users/sreekavya198@gmail.com/supply_chain_demand_forecast"

# COMMAND ----------


print("Loading feature-engineered data...")
features_spark_df = spark.table(GOLD_FEATURES)
print(f"Loaded {features_spark_df.count():,} records")

# Convert to Pandas for sklearn
print(" Converting to Pandas for ML training...")
features_pdf = features_spark_df.toPandas()
print(f"Converted to Pandas: {len(features_pdf):,} rows, {len(features_pdf.columns)} columns")


# COMMAND ----------


print(" Selecting features for ML model...")

# Define feature categories
lag_features = [
    "sales_lag_7", "sales_lag_14", "sales_lag_30", "sales_lag_60", "sales_lag_90"
]

rolling_features = [
    "rolling_mean_7", "rolling_mean_30", "rolling_mean_60",
    "rolling_std_7", "rolling_std_30"
]

trend_features = [
    "sales_growth_7d", "sales_growth_30d",
    "demand_volatility", "sales_momentum"
]

temporal_features = [
    "day_of_week", "month", "quarter", "is_weekend", "is_month_start", "is_month_end",
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"
]

supply_chain_features = [
    "competition_pressure", "sales_per_customer",
    "promo_frequency_30d", "assortment_score", "store_type_score"
]

business_flags = [
    "is_holiday", "is_promo_day", "Promo2", "has_competition"
]


# COMMAND ----------


# Combine all features
feature_cols = (lag_features + rolling_features + trend_features + 
                temporal_features + supply_chain_features + business_flags)

print(f"Selected {len(feature_cols)} features:")
print(f"  Lag features: {len(lag_features)}")
print(f"  Rolling features: {len(rolling_features)}")
print(f"  Trend features: {len(trend_features)}")
print(f"  Temporal features: {len(temporal_features)}")
print(f"  Supply chain features: {len(supply_chain_features)}")
print(f"  Business flags: {len(business_flags)}")


# COMMAND ----------


# Target variable
target = "Sales"

# Prepare X and y
X = features_pdf[feature_cols].fillna(0)
y = features_pdf[target]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")


# COMMAND ----------


print("Creating time-based train-test split...")

# Sort by date first
features_pdf_sorted = features_pdf.sort_values('Date')
X_sorted = features_pdf_sorted[feature_cols].fillna(0)
y_sorted = features_pdf_sorted[target]

# 80-20 split (chronological)
split_idx = int(len(X_sorted) * 0.8)

X_train = X_sorted[:split_idx]
X_test = X_sorted[split_idx:]
y_train = y_sorted[:split_idx]
y_test = y_sorted[split_idx:]

print(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X_sorted)*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X_sorted)*100:.1f}%)")


# COMMAND ----------

print(f"Setting up MLflow experiment: {EXPERIMENT_NAME}")
mlflow.set_experiment(EXPERIMENT_NAME)


# COMMAND ----------


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate comprehensive evaluation metrics"""
    
    # Filter out zero values for MAPE calculation
    mask = y_true != 0
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE:  ${metrics['mae']:,.2f}")
    print(f"  RMSE: ${metrics['rmse']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics

def plot_predictions(y_true, y_pred, model_name, sample_size=1000):
    """Create prediction vs actual plot"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sample for visualization
    indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
    y_true_sample = y_true.iloc[indices]
    y_pred_sample = y_pred[indices]
    
    # Scatter plot
    axes[0].scatter(y_true_sample, y_pred_sample, alpha=0.5)
    axes[0].plot([y_true_sample.min(), y_true_sample.max()], 
                 [y_true_sample.min(), y_true_sample.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual Sales')
    axes[0].set_ylabel('Predicted Sales')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    
    # Residual plot
    residuals = y_true_sample - y_pred_sample
    axes[1].scatter(y_pred_sample, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Sales')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name}: Residual Plot')
    
    plt.tight_layout()
    return fig


# COMMAND ----------


print("MODEL 1: BASELINE - MOVING AVERAGE")


# Use 30-day rolling mean as baseline prediction
y_pred_baseline = X_test['rolling_mean_30'].values

metrics_baseline = calculate_metrics(y_test, y_pred_baseline, "Baseline (Moving Average)")

with mlflow.start_run(run_name="baseline_moving_average"):
    mlflow.log_param("model_type", "baseline")
    mlflow.log_param("method", "30_day_moving_average")
    
    for metric_name, metric_value in metrics_baseline.items():
        mlflow.log_metric(metric_name, metric_value)

print("Baseline model evaluated and logged")


# COMMAND ----------


print("MODEL 2: LINEAR REGRESSION")


with mlflow.start_run(run_name="linear_regression"):
    
    print("Training Linear Regression...")
    
    # Log parameters
    mlflow.log_param("model_type", "linear_regression")
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("feature_list", ", ".join(feature_cols[:10]))  # First 10 features
    
    # Train model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    print("Model trained")
    
    # Predictions
    y_pred_lr_train = lr_model.predict(X_train)
    y_pred_lr_test = lr_model.predict(X_test)
    
    # Calculate metrics
    metrics_lr_train = calculate_metrics(y_train, y_pred_lr_train, "Linear Regression (Train)")
    metrics_lr_test = calculate_metrics(y_test, y_pred_lr_test, "Linear Regression (Test)")
    
    # Log metrics
    for metric_name, metric_value in metrics_lr_test.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    for metric_name, metric_value in metrics_lr_train.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lr_model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Log feature importance
    mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
    
    # Create and log visualization
    fig = plot_predictions(y_test, y_pred_lr_test, "Linear Regression")
    mlflow.log_figure(fig, "linear_regression_predictions.png")
    plt.close(fig)
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "linear_regression_model")
    
    print("Linear Regression model logged to MLflow")


# COMMAND ----------


print("MODEL 3: RANDOM FOREST REGRESSOR")


with mlflow.start_run(run_name="random_forest"):
    
    print("Training Random Forest...")
    
    # Hyperparameters
    rf_params = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Log parameters
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_params(rf_params)
    mlflow.log_param("n_features", len(feature_cols))
    
    # Train model
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train)
    
    print("Model trained")
    
    # Predictions
    y_pred_rf_train = rf_model.predict(X_train)
    y_pred_rf_test = rf_model.predict(X_test)
    
    # Calculate metrics
    metrics_rf_train = calculate_metrics(y_train, y_pred_rf_train, "Random Forest (Train)")
    metrics_rf_test = calculate_metrics(y_test, y_pred_rf_test, "Random Forest (Test)")
    
    # Log metrics
    for metric_name, metric_value in metrics_rf_test.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    for metric_name, metric_value in metrics_rf_train.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    
    # Feature importance
    feature_importance_rf = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_rf.head(10).to_string(index=False))
    
    # Log feature importance
    mlflow.log_text(feature_importance_rf.to_string(), "feature_importance.txt")
    
    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_rf.head(15)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances - Random Forest')
    ax.invert_yaxis()
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)
    
    # Create and log prediction visualization
    fig = plot_predictions(y_test, y_pred_rf_test, "Random Forest")
    mlflow.log_figure(fig, "random_forest_predictions.png")
    plt.close(fig)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    
    print("Random Forest model logged to MLflow")


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from xgboost import XGBRegressor
from mlflow.models.signature import infer_signature

print("MODEL 4: XGBOOST REGRESSOR (PRODUCTION MODEL)")

with mlflow.start_run(run_name="xgboost_production"):

    print("Training XGBoost...")

    # Hyperparameters (tuned)
    xgb_params = {
        'n_estimators': 150,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }

    # Log parameters
    mlflow.log_param("model_type", "xgboost")
    mlflow.log_params(xgb_params)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("feature_engineering_version", "v1.0")

    # Train model (NO early stopping – XGBoost 3.1.3 sklearn wrapper)
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train)

    print("Model trained")

    # Predictions
    y_pred_xgb_train = xgb_model.predict(X_train)
    y_pred_xgb_test = xgb_model.predict(X_test)

    # Calculate metrics
    metrics_xgb_train = calculate_metrics(
        y_train, y_pred_xgb_train, "XGBoost (Train)"
    )
    metrics_xgb_test = calculate_metrics(
        y_test, y_pred_xgb_test, "XGBoost (Test)"
    )

    # Log metrics
    for metric_name, metric_value in metrics_xgb_test.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)

    for metric_name, metric_value in metrics_xgb_train.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)

    # Feature importance
    feature_importance_xgb = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance_xgb.head(10).to_string(index=False))

    # Log feature importance table
    mlflow.log_text(
        feature_importance_xgb.to_string(index=False),
        "feature_importance.txt"
    )

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_xgb.head(15)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances - XGBoost')
    ax.invert_yaxis()
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

    # Prediction visualization
    fig = plot_predictions(y_test, y_pred_xgb_test, "XGBoost")
    mlflow.log_figure(fig, "xgboost_predictions.png")
    plt.close(fig)

    # Model signature
    signature = infer_signature(X_train, y_pred_xgb_train)

    # Log model ONCE (clean)
    mlflow.sklearn.log_model(
        xgb_model,
        "xgboost_model",
        signature=signature
    )

    print("XGBoost model logged to MLflow")


# COMMAND ----------

# display(feature_importance_xgb)

feature_importance_xgb_df = spark.createDataFrame(feature_importance_xgb)
feature_importance_xgb_df.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{GOLD_SCHEMA}.feature_importance_xgboost")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from supply_chain_catalog.gold_schema.feature_importance_xgboost

# COMMAND ----------


print("MODEL COMPARISON SUMMARY")


comparison_df = pd.DataFrame({
    'Model': ['Baseline', 'Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [
        metrics_baseline['mae'],
        metrics_lr_test['mae'],
        metrics_rf_test['mae'],
        metrics_xgb_test['mae']
    ],
    'RMSE': [
        metrics_baseline['rmse'],
        metrics_lr_test['rmse'],
        metrics_rf_test['rmse'],
        metrics_xgb_test['rmse']
    ],
    'R_square': [
        metrics_baseline['r2'],
        metrics_lr_test['r2'],
        metrics_rf_test['r2'],
        metrics_xgb_test['r2']
    ],
    'MAPE_percent': [
        metrics_baseline['mape'],
        metrics_lr_test['mape'],
        metrics_rf_test['mape'],
        metrics_xgb_test['mape']
    ]
})

print("\n" + comparison_df.to_string(index=False))


# COMMAND ----------

# Determine best model
best_model_idx = comparison_df['MAPE_percent'].idxmin()
best_model_name = comparison_df.loc[best_model_idx, 'Model']

print(f" BEST MODEL: {best_model_name}")
print(f"   MAPE: {comparison_df.loc[best_model_idx, 'MAPE_percent']:.2f}%")
print(f"   R²: {comparison_df.loc[best_model_idx, 'R_square']:.4f}")


# COMMAND ----------


# Save comparison
comparison_spark_df = spark.createDataFrame(comparison_df)
comparison_spark_df.write.format("delta").mode("overwrite") \
    .saveAsTable(f"{CATALOG}.{ML_SCHEMA}.model_comparison")

print("\nModel comparison saved to Delta")


# COMMAND ----------


print("SAVING PREDICTIONS TO DELTA")


# Create predictions DataFrame
test_data_with_predictions = features_pdf_sorted[split_idx:].copy()
test_data_with_predictions['predicted_sales_xgboost'] = y_pred_xgb_test
test_data_with_predictions['predicted_sales_rf'] = y_pred_rf_test
test_data_with_predictions['predicted_sales_lr'] = y_pred_lr_test
test_data_with_predictions['actual_sales'] = y_test.values

# Calculate errors
test_data_with_predictions['error_xgboost'] = (
    test_data_with_predictions['actual_sales'] - 
    test_data_with_predictions['predicted_sales_xgboost']
)
test_data_with_predictions['absolute_error_xgboost'] = abs(
    test_data_with_predictions['error_xgboost']
)
test_data_with_predictions['percentage_error_xgboost'] = (
    abs(test_data_with_predictions['error_xgboost']) / 
    test_data_with_predictions['actual_sales'] * 100
)

# Select columns for saving
predictions_cols = [
    'Store', 'Date', 'actual_sales',
    'predicted_sales_xgboost', 'predicted_sales_rf', 'predicted_sales_lr',
    'error_xgboost', 'absolute_error_xgboost', 'percentage_error_xgboost'
]

predictions_df = test_data_with_predictions[predictions_cols]


# COMMAND ----------


# Convert to Spark and save
predictions_spark_df = spark.createDataFrame(predictions_df)

predictions_spark_df.write.format("delta").mode("overwrite") \
    .saveAsTable(ML_PREDICTIONS)

# Add table comment
spark.sql(f"""
    COMMENT ON TABLE {ML_PREDICTIONS} IS
    'ML model predictions for demand forecasting.
    Contains predictions from XGBoost (production), Random Forest, and Linear Regression.
    Includes actual sales, errors, and percentage errors for evaluation.'
""")

print(f"Saved {len(predictions_df):,} predictions to {ML_PREDICTIONS}")


# COMMAND ----------


print("ML MODEL TRAINING COMPLETE!")


print(f"Trained and evaluated 4 models:")
print(f"   Baseline (Moving Average)")
print(f"   Linear Regression")
print(f"   Random Forest")
print(f"   XGBoost (Production Model)")

print(f"Best Model: {best_model_name}")
print(f"   MAPE: {comparison_df.loc[best_model_idx, 'MAPE_percent']:.2f}%")
print(f"   Target: <15% (Achieved: {'YES' if comparison_df.loc[best_model_idx, 'MAPE_percent'] < 15 else '✗ NO'})")

print(f"All models logged to MLflow")
print(f" Predictions saved to Delta")
print(f" Model comparison table created")

print(f" Ready for inventory optimization on Day 5!")

# Display sample predictions
print("Sample Predictions:")
display(spark.table(ML_PREDICTIONS).limit(10))

# COMMAND ----------

