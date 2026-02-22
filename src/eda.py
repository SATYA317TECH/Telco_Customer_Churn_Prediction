import pandas as pd
import numpy as np
from src.data_ingestion import load_churn_data

df = load_churn_data()

def data_health_check(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sort_values(ascending=False)
    }

health = data_health_check(df)
print(health)

churn_dist = df["churn"].value_counts(normalize=True)
print(churn_dist)

tenure_churn = df.groupby("churn")["tenure_months"].mean()
print(tenure_churn)

contract_churn = (
    df.groupby("contract_type")["churn"]
      .mean()
      .sort_values(ascending=False)
)
print(contract_churn)

billing_churn = df.groupby("churn")[[
    "monthly_charges",
    "late_payments"
]].mean()
print(billing_churn)

payment_churn = (
    df.groupby("payment_method")["churn"]
      .mean()
      .sort_values(ascending=False)
)
print(payment_churn)

usage_churn = df.groupby("churn")[[
    "avg_call_minutes",
    "avg_data_usage_gb"
]].mean()
print(usage_churn)

cx_churn = df.groupby("churn")[[
    "support_ticket_count",
    "avg_resolution_time",
    "avg_satisfaction_score"
]].mean()
print(cx_churn)

addons_churn = df.groupby("churn")[[
    "has_online_security",
    "has_tech_support",
    "streaming_services_count"
]].mean()
print(addons_churn)

with open("logs/eda_summary.txt", "w") as f:
    f.write("CHURN DISTRIBUTION\n")
    f.write(str(churn_dist))
    f.write("\n\nTENURE VS CHURN\n")
    f.write(str(tenure_churn))