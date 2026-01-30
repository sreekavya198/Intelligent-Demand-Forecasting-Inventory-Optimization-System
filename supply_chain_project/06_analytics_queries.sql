-- Databricks notebook source

-- This query provides high-level KPIs for executive dashboard
CREATE OR REPLACE VIEW supply_chain_catalog.analytics_schema.kpi_summary AS
SELECT
    COUNT(DISTINCT Store) as total_stores,
    AVG(percentage_error_xgboost) as avg_forecast_accuracy_pct,
    (100 - AVG(percentage_error_xgboost)) as forecast_accuracy_score,
    SUM(actual_sales) as total_actual_sales,
    SUM(predicted_sales_xgboost) as total_predicted_sales,
    SUM(ABS(error_xgboost)) as total_absolute_error,
    CURRENT_DATE() as report_date
FROM supply_chain_catalog.ml_schema.demand_predictions;


-- COMMAND ----------


-- View KPIs
SELECT 
    'Total Stores' as metric,
    CAST(total_stores AS STRING) as value
FROM supply_chain_catalog.analytics_schema.kpi_summary
UNION ALL
SELECT 
    'Forecast Accuracy',
    CONCAT(ROUND(forecast_accuracy_score, 2), '%')
FROM supply_chain_catalog.analytics_schema.kpi_summary
UNION ALL
SELECT
    'Total Sales (Test Period)',
    CONCAT('$', FORMAT_NUMBER(total_actual_sales, 2))
FROM supply_chain_catalog.analytics_schema.kpi_summary;


-- COMMAND ----------

-- ============================================
-- QUERY 2: TOP 20 MOST ACCURATE FORECASTS
-- ============================================

SELECT
    Store,
    COUNT(*) as prediction_count,
    AVG(predicted_sales_xgboost) as avg_predicted_sales,
    AVG(actual_sales) as avg_actual_sales,
    AVG(percentage_error_xgboost) as avg_mape,
    (100 - AVG(percentage_error_xgboost)) as accuracy_score
FROM supply_chain_catalog.ml_schema.demand_predictions
WHERE actual_sales > 0
GROUP BY Store
ORDER BY avg_mape ASC
LIMIT 20;

-- COMMAND ----------


-- ============================================
-- QUERY 3: FORECAST ACCURACY BY STORE TYPE
-- ============================================

CREATE OR REPLACE VIEW supply_chain_catalog.analytics_schema.accuracy_by_store_type AS
SELECT
    s.StoreType,
    s.Assortment,
    COUNT(DISTINCT p.Store) as store_count,
    AVG(p.percentage_error_xgboost) as avg_mape,
    AVG(p.actual_sales) as avg_actual_sales,
    AVG(p.predicted_sales_xgboost) as avg_predicted_sales,
    SUM(p.actual_sales) as total_actual_sales,
    SUM(p.predicted_sales_xgboost) as total_predicted_sales
FROM supply_chain_catalog.ml_schema.demand_predictions p
JOIN supply_chain_catalog.silver_schema.cleaned_stores s
    ON p.Store = s.Store
GROUP BY s.StoreType, s.Assortment
ORDER BY avg_mape ASC;

SELECT * FROM supply_chain_catalog.analytics_schema.accuracy_by_store_type;


-- COMMAND ----------


-- ============================================
-- QUERY 4: HIGH-RISK STOCKOUT ALERTS
-- ============================================

SELECT
    Store,
    current_inventory,
    reorder_point,
    optimal_order_qty,
    ROUND(days_until_stockout, 1) as days_until_stockout,
    ROUND((reorder_point - current_inventory), 0) as inventory_gap,
    recommended_action,
    ROUND(forecast_next_30d, 0) as forecast_next_30d
FROM supply_chain_catalog.ml_schema.inventory_recommendations
WHERE stockout_risk = 'HIGH'
ORDER BY days_until_stockout ASC
LIMIT 50;


-- COMMAND ----------


-- ============================================
-- QUERY 5: INVENTORY OPTIMIZATION SAVINGS BY STORE
-- ============================================

CREATE OR REPLACE VIEW supply_chain_catalog.analytics_schema.savings_by_store AS
SELECT
    Store,
    ROUND(current_inventory, 0) as current_inventory,
    ROUND(optimal_order_qty, 0) as optimal_inventory,
    ROUND(current_inventory - optimal_order_qty, 0) as inventory_reduction,
    ROUND((current_inventory - optimal_order_qty) * 2.50, 2) as monthly_savings,
    ROUND((current_inventory - optimal_order_qty) * 2.50 * 12, 2) as annual_savings,
    stockout_risk
FROM supply_chain_catalog.ml_schema.inventory_recommendations
WHERE (current_inventory - optimal_order_qty) > 0
ORDER BY annual_savings DESC;

SELECT 
    SUM(annual_savings) as total_annual_savings,
    AVG(annual_savings) as avg_savings_per_store,
    COUNT(*) as stores_with_savings
FROM supply_chain_catalog.analytics_schema.savings_by_store;


-- COMMAND ----------


-- ============================================
-- QUERY 6: DEMAND TREND ANALYSIS
-- ============================================

SELECT
    DATE_TRUNC('week', Date) as week_start,
    SUM(actual_sales) as total_actual_sales,
    SUM(predicted_sales_xgboost) as total_predicted_sales,
    AVG(percentage_error_xgboost) as weekly_mape,
    COUNT(*) as prediction_count
FROM supply_chain_catalog.ml_schema.demand_predictions
GROUP BY DATE_TRUNC('week', Date)
ORDER BY week_start;


-- COMMAND ----------

SELECT
    feature,
    importance,
    RANK() OVER (ORDER BY importance DESC) as importance_rank
FROM supply_chain_catalog.gold_schema.feature_importance_xgboost
ORDER BY importance DESC
LIMIT 15;

-- COMMAND ----------


-- ============================================
-- QUERY 8: PROMO EFFECTIVENESS ANALYSIS
-- ============================================

CREATE OR REPLACE VIEW supply_chain_catalog.analytics_schema.promo_effectiveness AS
SELECT
    CASE WHEN f.Promo = 1 THEN 'With Promotion' ELSE 'No Promotion' END as promo_status,
    COUNT(*) as days_count,
    AVG(p.actual_sales) as avg_daily_sales,
    AVG(p.predicted_sales_xgboost) as avg_predicted_sales,
    AVG(f.sales_per_customer) as avg_sales_per_customer
FROM supply_chain_catalog.ml_schema.demand_predictions p
JOIN supply_chain_catalog.gold_schema.demand_features f
    ON p.Store = f.Store AND p.Date = f.Date
WHERE p.actual_sales > 0
GROUP BY f.Promo;

SELECT * FROM supply_chain_catalog.analytics_schema.promo_effectiveness;


-- COMMAND ----------


-- ============================================
-- QUERY 9: COMPETITION IMPACT ANALYSIS
-- ============================================

SELECT
    CASE
        WHEN s.CompetitionDistance < 1000 THEN 'Very Close (<1km)'
        WHEN s.CompetitionDistance < 5000 THEN 'Close (1-5km)'
        WHEN s.CompetitionDistance < 10000 THEN 'Moderate (5-10km)'
        ELSE 'Far (>10km)'
    END as competition_proximity,
    COUNT(DISTINCT p.Store) as store_count,
    AVG(p.actual_sales) as avg_daily_sales,
    AVG(p.percentage_error_xgboost) as avg_forecast_error
FROM supply_chain_catalog.ml_schema.demand_predictions p
JOIN supply_chain_catalog.silver_schema.cleaned_stores s
    ON p.Store = s.Store
GROUP BY 
    CASE
        WHEN s.CompetitionDistance < 1000 THEN 'Very Close (<1km)'
        WHEN s.CompetitionDistance < 5000 THEN 'Close (1-5km)'
        WHEN s.CompetitionDistance < 10000 THEN 'Moderate (5-10km)'
        ELSE 'Far (>10km)'
    END
ORDER BY avg_daily_sales DESC;


-- COMMAND ----------


-- ============================================
-- QUERY 10: BUSINESS IMPACT SUMMARY
-- ============================================

SELECT
    analysis_date,
    total_stores,
    ROUND(forecasted_demand_30d, 0) as forecasted_demand,
    ROUND(inventory_reduction_units, 0) as inventory_reduced_units,
    ROUND(inventory_reduction_pct, 1) as inventory_reduced_pct,
    CONCAT('$', FORMAT_NUMBER(monthly_holding_cost_savings, 2)) as monthly_savings,
    CONCAT('$', FORMAT_NUMBER(annual_holding_cost_savings, 2)) as annual_savings,
    high_risk_stores,
    ROUND(stockouts_prevented, 0) as stockouts_prevented,
    CONCAT('$', FORMAT_NUMBER(stockout_cost_savings, 2)) as stockout_savings,
    CONCAT('$', FORMAT_NUMBER(total_annual_roi, 2)) as total_annual_roi
FROM supply_chain_catalog.ml_schema.business_impact_analysis
ORDER BY analysis_date DESC
LIMIT 1;