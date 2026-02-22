import pandas as pd
import numpy as np
from src.data_ingestion import load_churn_data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # 1. TARGET CHECK
    # -----------------------------
    df = df[df["churn"].isin([0, 1])]

    # -----------------------------
    # 2. HANDLE MISSING VALUES
    # -----------------------------

    # Customer experience (missing = no interaction)
    cx_cols = [
        "support_ticket_count",
        "avg_resolution_time",
        "avg_satisfaction_score"
    ]
    df[cx_cols] = df[cx_cols].fillna(0)

    # Usage (missing = no usage)
    usage_cols = ["avg_call_minutes", "avg_data_usage_gb"]
    df[usage_cols] = df[usage_cols].fillna(0)

    # Billing
    df["late_payments"] = df["late_payments"].fillna(0)

    # total_charges: missing usually means very low tenure
    df["total_charges"] = df["total_charges"].fillna(
        df["monthly_charges"] * df["tenure_months"]
    )

    # -----------------------------
    # 3. CLEAN CATEGORICAL STRINGS
    # -----------------------------
    cat_cols = [
        "contract_type",
        "payment_method"
    ]

    for col in cat_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # -----------------------------
    # 4. OUTLIER CAPPING (SAFE)
    # -----------------------------
    def cap_outliers(series, lower_q=0.01, upper_q=0.99):
        lower = series.quantile(lower_q)
        upper = series.quantile(upper_q)
        return series.clip(lower, upper)

    num_cols = [
        "monthly_charges",
        "total_charges",
        "avg_call_minutes",
        "avg_data_usage_gb",
        "support_ticket_count",
        "late_payments"
    ]

    for col in num_cols:
        df[col] = cap_outliers(df[col])

    return df


if __name__ == "__main__":
    df = load_churn_data()
    df_clean = clean_data(df)
    print(df_clean.info())