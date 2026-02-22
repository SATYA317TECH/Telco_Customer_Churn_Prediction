from src.feature_engineering import engineer_features
from src.data_cleaning import clean_data
from src.data_ingestion import load_churn_data

import pandas as pd

df = engineer_features(clean_data(load_churn_data()))

# -----------------------------
# TARGET SEPARATION CHECK
# -----------------------------
print(df.groupby("churn")[[
    "engagement_score",
    "cx_risk_score",
    "stickiness_score",
    "charges_per_month"
]].mean())

# -----------------------------
# FEATURE VARIANCE CHECK
# -----------------------------
low_variance = df.nunique()
print(low_variance.sort_values())

# -----------------------------
# TENURE BUCKET VS CHURN
# -----------------------------
print(df.groupby("tenure_bucket")["churn"].mean())