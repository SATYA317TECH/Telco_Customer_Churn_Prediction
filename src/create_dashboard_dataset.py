import pandas as pd
import joblib

from src.data_ingestion import load_churn_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features


MODEL_PATH = "artifacts/churn_model_v1.joblib"
THRESHOLD = 0.40


def create_dashboard_dataset():
    df = engineer_features(clean_data(load_churn_data()))

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]

    df["churn_probability"] = model.predict_proba(df)[:, 1]
    df["churn_flag"] = (df["churn_probability"] >= THRESHOLD).astype(int)

    df["risk_segment"] = pd.cut(
        df["churn_probability"],
        bins=[0, 0.25, 0.40, 1.0],
        labels=["Low", "Medium", "High"]
    )

    # Action category (high-level)
    df["action_category"] = df["risk_segment"].map({
        "High": "Apply retention strategy",
        "Medium": "Monitor customer",
        "Low": "No action required"
    })

    dashboard_df = df[
        [
            "customer_id",
            "churn_probability",
            "churn_flag",
            "risk_segment",
            "monthly_charges",
            "tenure_months",
            "contract_type",
            "cx_risk_score",
            "payment_method",
            "stickiness_score",
            "action_category"
        ]
    ]

    dashboard_df.to_csv(
        "dashboard/churn_dashboard_dataset.csv",
        index=False
    )

    print("Dashboard dataset created successfully!")


if __name__ == "__main__":
    create_dashboard_dataset()
