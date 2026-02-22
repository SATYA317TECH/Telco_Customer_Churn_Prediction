import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.feature_engineering import engineer_features
from src.data_cleaning import clean_data
from src.data_ingestion import load_churn_data

def encode_and_scale():
    
    print("\n" + "="*60)
    print("ENCODING & SCALING - PREPARING FEATURES")
    print("="*60)

    # Load → Clean → Engineer
    print("\n📥 Loading and engineering features...")
    df = engineer_features(clean_data(load_churn_data()))
    print(f"✅ Total features available: {df.shape[1]} columns")

    # -----------------------------
    # TARGET
    # -----------------------------
    y = df["churn"]
    print(f"\n🎯 Target distribution:\n{y.value_counts(normalize=True).mul(100).round(2)}")

    # -----------------------------
    # FEATURE GROUPS
    # -----------------------------
    numerical_features = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "late_payments",
        "avg_call_minutes",
        "avg_data_usage_gb",
        "support_ticket_count",
        "avg_resolution_time",
        "avg_satisfaction_score",
        "charges_per_month",
        "engagement_score",
        "cx_risk_score",
        "stickiness_score"
    ]

    categorical_features = [
        "contract_type",
        "payment_method",
        "tenure_bucket"
    ]

    X = df[numerical_features + categorical_features]

    # ============================================================================
    # PREPROCESSOR (ENCODING + SCALING)
    # ============================================================================
    print("\n" + "="*60)
    print("CREATING PREPROCESSOR")
    print("="*60)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    features = numerical_features + categorical_features
    print("\n📊 Total input features selected:", len(features))

    return X, y, preprocessor, features

if __name__ == "__main__":
    X, y, preprocessor, feature_names = encode_and_scale()
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts(normalize=True))
    print("\n✅ Encoding and scaling setup complete!")


    