import joblib
import pandas as pd

MODEL_PATH = "artifacts/churn_model_v1.joblib"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_churn(sample_data: dict, verbose=True):
    bundle = load_model()
    model = bundle["model"]
    threshold = bundle["threshold"]

    df = pd.DataFrame([sample_data])

    churn_prob = float(model.predict_proba(df)[:, 1][0])
    churn_pred = int(churn_prob >= threshold)
    
    # Determine risk level
    if churn_prob >= 0.7:
        risk_level = "HIGH"
        risk_description = "Immediate retention action required"
    elif churn_prob >= 0.4:
        risk_level = "MEDIUM"
        risk_description = "Monitor closely, proactive outreach recommended"
    else:
        risk_level = "LOW"
        risk_description = "No immediate action needed"

    result = {
        "churn_probability": round(churn_prob, 4),
        "churn_probability_pct": f"{round(churn_prob * 100, 2)}%",
        "churn_prediction": churn_pred,
        "risk_level": risk_level,
        "risk_description": risk_description,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"PREDICTION RESULT")
        print(f"{'='*50}")
        print(f"Churn Probability: {result['churn_probability_pct']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Risk Description: {result['risk_description']}")
        print(f"{'='*50}")

    return result


if __name__ == "__main__":

    print("\n" + "="*60)
    print("TESTING CHURN PREDICTION MODEL")
    print("="*60)
    
    # -----------------------------
    # Sample Customer 1
    # -----------------------------
    sample_customer_1 = {
        "tenure_months": 3,
        "monthly_charges": 80,
        "total_charges": 250,
        "late_payments": 2,
        "avg_call_minutes": 120,
        "avg_data_usage_gb": 5,
        "support_ticket_count": 3,
        "avg_resolution_time": 72,
        "avg_satisfaction_score": 2,
        "charges_per_month": 83,
        "engagement_score": 150,
        "cx_risk_score": 9,
        "stickiness_score": 0,
        "contract_type": "month-to-month",
        "payment_method": "electronic check",
        "tenure_bucket": "0-6"
    }
    
    # -----------------------------
    # Sample Customer 2
    # -----------------------------
    sample_customer_2 = {
        "tenure_months": 10,
        "monthly_charges": 75,
        "total_charges": 1125,
        "late_payments": 1,
        "avg_call_minutes": 200,
        "avg_data_usage_gb": 12,
        "support_ticket_count": 2,
        "avg_resolution_time": 48,
        "avg_satisfaction_score": 3,
        "charges_per_month": 75,
        "engagement_score": 200,
        "cx_risk_score": 4,
        "stickiness_score": 1,
        "contract_type": "month-to-month",
        "payment_method": "electronic check",
        "tenure_bucket": "12-24"
    }

    # -----------------------------
    # Sample Customer 3
    # -----------------------------
    sample_customer_3 = {
        "tenure_months": 36,
        "monthly_charges": 50,
        "total_charges": 4000,
        "late_payments": 1,
        "avg_call_minutes": 500,
        "avg_data_usage_gb": 30,
        "support_ticket_count": 1,
        "avg_resolution_time": 24,
        "avg_satisfaction_score": 5,
        "charges_per_month": 30,
        "engagement_score": 500,
        "cx_risk_score": 0,
        "stickiness_score": 3,
        "contract_type": "one year",
        "payment_method": "credit card",
        "tenure_bucket": "24-48"
    }
    
    # Test all three customers
    print("\nTesting Sample Customer 1:")
    result = predict_churn(sample_customer_1, verbose=True)
    print("\nTesting Sample Customer 2:")
    result = predict_churn(sample_customer_2, verbose=True)
    print("\nTesting Sample Customer 3:")
    result = predict_churn(sample_customer_3, verbose=True)
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    
    results = []
    for customer in [
        sample_customer_1,
        sample_customer_2,
        sample_customer_3
    ]:
        res = predict_churn(customer, verbose=False)
        if res['churn_probability'] > 0.7:
            name="High Risk Customer"
        elif res['churn_probability'] > 0.4:
            name="Medium Risk Customer"
        else:
            name="Low Risk Customer"
        results.append({
            "Customer Type": name,
            "Probability": res['churn_probability_pct'],
            "Risk Level": res['risk_level'],
            "Action": res['risk_description']
        })
    
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))