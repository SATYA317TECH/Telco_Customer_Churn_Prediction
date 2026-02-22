import pandas as pd
import numpy as np
from src.data_cleaning import clean_data
from src.data_ingestion import load_churn_data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # 1. TENURE BUCKETS (LIFECYCLE)
    # -----------------------------
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 12, 24, 48, 1000],
        labels=["0-6", "6-12", "12-24", "24-48", "48+"]
    )

    # -----------------------------
    # 2. PRICE SENSITIVITY
    # -----------------------------
    df["charges_per_month"] = (
        df["total_charges"] / (df["tenure_months"] + 1)
    )

    df["high_price_flag"] = (
        df["monthly_charges"] > df["monthly_charges"].median()
    ).astype(int)

    # -----------------------------
    # 3. ENGAGEMENT SCORE
    # -----------------------------
    df["engagement_score"] = (
        0.4 * df["avg_call_minutes"] +
        0.6 * df["avg_data_usage_gb"]
    )

    # -----------------------------
    # 4. CUSTOMER EXPERIENCE SCORE
    # -----------------------------
    df["cx_risk_score"] = (
        df["support_ticket_count"] *
        (5 - df["avg_satisfaction_score"])
    )

    # -----------------------------
    # 5. PAYMENT RISK
    # -----------------------------
    df["payment_risk"] = (
        df["late_payments"] > 0
    ).astype(int)

    # -----------------------------
    # 6. STICKINESS SCORE
    # -----------------------------
    df["stickiness_score"] = (
        df["has_online_security"] +
        df["has_tech_support"] +
        df["streaming_services_count"]
    )

    return df


if __name__ == "__main__":
    raw_df = load_churn_data()
    clean_df = clean_data(raw_df)
    fe_df = engineer_features(clean_df)
    print(fe_df.head())