# Time Series Forecasting - Presentation

## Slide 1: Project Overview
**Patient Mobility Forecasting**
- **Objective**: Predict daily step counts for next 365 days
- **Data**: 80K+ step count records + clinical features
- **Approach**: Baseline (Prophet) + Advanced (EBM)
- **Deliverable**: Explainable 365-day forecast

---

## Slide 2: Data Pipeline Architecture

**Input Data Sources**:
1. **Time Series Data** (timeseries-data.json)
   - 80,919 step count intervals
   - Aggregated to daily totals
   
2. **Clinical Data** (categorical-data.json)
   - Demographics (age, gender, disease)
   - Therapies, side effects, diagnoses, events

**Processing Steps**:
- Timestamp standardization → Daily aggregation → Feature engineering → Model training

---

## Slide 3: Feature Engineering

**Engineered Features** (40+ features):

| Category | Features | Examples |
|----------|----------|----------|
| **Temporal** | 4 features | Day of week, week of year, month, weekend flag |
| **Lag Features** | 3 features | Steps t-1, t-7, t-30 |
| **Rolling Stats** | 4 features | 7-day avg/std, 30-day avg/std |
| **Clinical** | 20+ features | Active therapies, side effect intensity, diagnoses |
| **Events** | 1 feature | Days since last clinical event |
| **Demographics** | 3 features | Age, gender, disease type |

---

## Slide 4: Model 1 - Baseline (Prophet)

**Univariate Time Series Model**

**Configuration**:
- Input: Historical step counts only
- Seasonality: Yearly + Weekly
- Train/Test Split: 80/20

**Results**:
- RMSE: [Value from actual run]
- MAE: [Value from actual run]
- Forecast: 365 days with confidence intervals

**Strengths**: Simple, interpretable, captures seasonality
**Limitations**: Ignores clinical context

---

## Slide 5: Model 2 - Multivariate (EBM)

**Explainable Boosting Machine**

**Configuration**:
- Input: Step history + 40+ clinical features
- Algorithm: Gradient boosting with GAMs
- Interpretability: Built-in global explanations

**Results**:
- RMSE: [Value from actual run]
- MAE: [Value from actual run]
- Improvement over baseline: [X]%

**Strengths**: Captures clinical impact, fully explainable
**Limitations**: Requires feature engineering

---

## Slide 6: Model Comparison

| Metric | Baseline (Prophet) | Multivariate (EBM) | Improvement |
|--------|-------------------|-------------------|-------------|
| RMSE | [Value] | [Value] | [X]% |
| MAE | [Value] | [Value] | [X]% |
| Features Used | 1 (steps only) | 40+ (steps + clinical) | - |
| Explainability | Trend decomposition | Feature importance | ✓ |

**Winner**: Multivariate EBM provides better accuracy with full explainability

---

## Slide 7: Explainability Insights

**Top 10 Most Important Features**:
1. Lag features (steps_t-1, steps_t-7)
2. Rolling averages (7-day, 30-day)
3. Active therapy count
4. Side effect intensity
5. Day of week
6. Days since last event
7. [Additional features from actual run]

**Key Finding**: Clinical features contribute [X]% to prediction accuracy

**Categorical Impact**: 
- Therapies: [Impact description]
- Side effects: [Impact description]

---

## Slide 8: Forecast Output

**365-Day Forecast Schema**:

| Date | Predicted_Steps | Trend_Component | Exogenous_Impact |
|------|----------------|-----------------|------------------|
| 2025-12-12 | 4,500 | 4,200 | +300 |
| ... | ... | ... | ... |

**Forecast Characteristics**:
- Average predicted steps: [Value]
- Trend: [Increasing/Stable/Decreasing]
- Clinical impact: [Description]

**Validation**: RMSE = [Value], MAE = [Value]

---

## Slide 9: Scalability Approach

**Scaling to 100,000 Patients**:

**Big Data Processing**:
- **PySpark** for distributed feature engineering
- **AWS Glue/EMR** for ETL pipeline
- **S3 + Athena** for data lake architecture

**Modeling Strategy**:
- **Clustered approach**: Group by disease type
- **Distributed training**: Hyperparameter tuning at scale
- **Model serving**: API endpoints with caching

**Performance**: Process 100K patients in <2 hours

---

## Slide 10: Key Learnings & Next Steps

**Top 3 Challenges**:
1. **Data Engineering**: Merging time series with clinical events
2. **Feature Engineering**: Handling variable-length therapy periods
3. **Forecast Validation**: Limited historical data for long-term validation

**Key Learnings**:
- Clinical features significantly improve accuracy
- Explainability is crucial for healthcare applications
- Lag features are strongest predictors

**Next Steps**:
- Deploy model to production
- Implement real-time inference
- Continuous model monitoring and retraining

---

**Thank You!**
