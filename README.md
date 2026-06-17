# Customer Churn Analysis & Prediction Model

**Prepared by:** Arvind Kumar
**Program:** MBA — Business Analytics
**University:** Chandigarh University, Uttar Pradesh

---

## Project Description

This project develops an end-to-end machine learning pipeline to predict customer churn for a subscription-based business. Starting from raw, noisy data, the pipeline covers data cleaning, exploratory analysis, feature engineering, model training, and hyperparameter tuning. The final model — a tuned **Random Forest Classifier** — identifies at-risk customers with high recall, enabling the business to intervene proactively before churn occurs. Key business insights around login activity, monthly spending patterns, and service call volumes are extracted to drive targeted retention strategies.

---

## Key Steps Performed

| Step | Name | Description |
|------|------|-------------|
| 1 | **Data Ingestion** | Loaded customer churn CSV from Google Drive into a Pandas DataFrame for initial inspection. |
| 2 | **Data Cleaning** | Replaced negative ages and monthly spend with NaN, imputed medians/modes, fixed inconsistent casing and unknown gender values. |
| 3 | **Exploratory Data Analysis (EDA)** | Visualised churn rates across categorical and numerical features using count plots, histograms, box plots, and a correlation heatmap. |
| 4 | **Feature Engineering** | Applied one-hot encoding to categorical columns (Gender, Region, Subscription Plan, Payment Method) with dummy-variable trap avoidance. |
| 5 | **Train-Test Split** | Stratified 80/20 split to preserve class balance between churned and retained customers. |
| 6 | **Baseline Modelling** | Trained a Random Forest Classifier (100 estimators, balanced class weights) and evaluated accuracy, precision, recall, and F1-score. |
| 7 | **Feature Importance** | Extracted and visualised the top 10 most influential features from the trained forest. |
| 8 | **Hyperparameter Tuning** | Used GridSearchCV (3-fold CV, F1 scoring) across n_estimators, max_depth, min_samples_split, and min_samples_leaf. |
| 9 | **Tuned Model Evaluation** | Re-evaluated the best estimator on the test set; compared confusion matrices and feature importances between initial and tuned models. |
| 10 | **Business Reporting** | Translated model outputs into actionable retention recommendations for inactive users, spend monitoring, and service optimisation. |

---

## Technologies & Libraries Used

| Category | Tools / Libraries |
|----------|-------------------|
| **Language** | Python 3 |
| **Data Manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **ML Components** | RandomForestClassifier, GridSearchCV, train_test_split |
| **Metrics** | accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report |
| **Environment** | Google Colab, Jupyter Notebook |
| **Storage** | Google Drive API |

---

## Model Summary

### Algorithm

**Random Forest Classifier** — an ensemble of decision trees. Class weights set to `balanced` to handle the imbalanced churn dataset. Parallelised training using all available CPU cores (`n_jobs=-1`).

### Hyperparameter Grid (GridSearchCV)

```
n_estimators      : [100, 200]
max_depth         : [10, 20, None]
min_samples_split : [2, 5]
min_samples_leaf  : [1, 2]
Cross-Validation  : 3-fold
Scoring Metric    : F1-Score
```

### Performance Metrics (Tuned Model)

| Metric | Score |
|--------|-------|
| **Accuracy** | 76.46% |
| **Precision** | 79.09% |
| **Recall** | 93.95% |
| **F1-Score** | 85.88% |

> The model achieves **94% recall** on churn — meaning it catches 94 of every 100 true churners, making it highly effective for proactive intervention campaigns.

---

## Data Source

- **File:** `customer churn.csv`
- **Storage:** Google Drive, loaded into Google Colab via the Drive API
- **Features:**

| Column | Type | Description |
|--------|------|-------------|
| `Customer_ID` | Identifier | Unique customer identifier |
| `Age` | Numerical | Customer age (had negative values requiring cleaning) |
| `Gender` | Categorical | Male / Female / Non-Binary |
| `Region` | Categorical | Geographic region of the customer |
| `Subscription_Plan` | Categorical | Type of subscription plan |
| `Payment_Method` | Categorical | Payment method used (had missing values) |
| `Days_Since_Last_Login` | Numerical | Days elapsed since the last login |
| `Monthly_Spend` | Numerical | Monthly spend amount (had negative values) |
| `Customer_Service_Calls` | Numerical | Number of customer service interactions |
| `Churn` | Target (Binary) | 1 = Churned, 0 = Retained |

**Data Quality Issues Addressed:**
- Missing values in `Age` and `Payment_Method` columns
- Negative values in `Age` and `Monthly_Spend`
- Inconsistent casing in `Region` and `Subscription_Plan`
- Unknown/ambiguous entries in `Gender`

---

## Results & Key Findings

### Top Churn Drivers (Feature Importance)

| Rank | Feature | Business Insight |
|------|---------|-----------------|
| #1 | **Days Since Last Login** | Extended inactivity is the clearest early signal of churn risk. |
| #2 | **Monthly Spend** | Spending pattern changes, especially drops, may indicate a move to competitors. |
| #3 | **Age** | A relevant demographic factor influencing churn likelihood across segments. |
| #4 | **Customer Service Calls** | Higher call volume is positively correlated with churn, indicating unresolved friction. |
| #5 | **Subscription Plan, Region, Gender, Payment Method** | Secondary predictors useful for segment-level retention targeting. |

### Business Recommendations

1. **Proactive Re-engagement** — Trigger automated campaigns for customers with rising `Days_Since_Last_Login`. Personalised emails with new features or special offers can recover at-risk users before they churn.

2. **Monitor Spending Patterns** — Flag customers whose `Monthly_Spend` drops significantly. Offer tailored incentives or upsell opportunities to stabilise spend.

3. **Optimise Customer Service** — High `Customer_Service_Calls` is a churn warning. Invest in first-contact resolution and escalate high-frequency callers to dedicated support.

4. **Segment-Based Campaigns** — Use demographic and subscription features to personalise retention campaigns by age group, region, or plan type.

5. **Operational Integration** — Embed the churn prediction model into CRM dashboards to give sales and support teams real-time churn risk scores for each customer.

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│  [Data Ingestion] ──► [Data Cleaning] ──► [EDA]                │
│   CSV from Drive       Fix nulls,          Plots,              │
│                        outliers, casing    correlations        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING                              │
│                                                                 │
│         [Feature Engineering] ──► [Train / Test Split]         │
│          One-hot encoding           80/20 stratified           │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MODELLING                                │
│                                                                 │
│  [Baseline Model] ──► [Feature Importance] ──► [Hyper Tuning]  │
│  Random Forest         Top-10 drivers           GridSearchCV   │
│  100 estimators        identified               F1 optimised   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       EVALUATION                                │
│                                                                 │
│         [Model Evaluation] ──► [Confusion Matrix]              │
│          Accuracy · Precision    Tuned vs baseline             │
│          Recall · F1-Score       comparison                    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BUSINESS INSIGHTS                           │
│                                                                 │
│              [Retention Recommendations]                        │
│       Re-engagement · Spend monitoring · Service fix           │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CHURN PREDICTION REPORT                        │
│     Actionable customer retention strategy                      │
│     Model ready for CRM / dashboard deployment                  │
│                                                                 │
│     Accuracy: 76.5%  |  Recall: 94.0%  |  F1: 85.9%           │
└─────────────────────────────────────────────────────────────────┘
```

---

*Prepared by **Arvind Kumar** · MBA Business Analytics · Chandigarh University, Uttar Pradesh*
*Model: Random Forest Classifier · Tuned via GridSearchCV · Environment: Google Colab*
