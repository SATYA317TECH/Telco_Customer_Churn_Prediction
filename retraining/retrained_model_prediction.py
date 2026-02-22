import pandas as pd
import joblib
from retraining.model_retraining import MODEL_PATH


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
    print("TESTING RETRAINED DEPLOYMENT MODEL (7 FEATURES)")
    print("="*60)
    
    # -----------------------------
    # Sample Customer 1
    # -----------------------------
    sample_customer_1 = {
        "tenure_months": 3,
        "contract_type": "month-to-month",
        "monthly_charges": 110,
        "payment_method": "electronic check",
        "support_ticket_count": 2,
        "avg_call_minutes": 120,
        "avg_data_usage_gb": 5
    }
    
    # -----------------------------
    # Sample Customer 2
    # -----------------------------
    sample_customer_2 = {
        "tenure_months": 20,
        "contract_type": "one year",
        "monthly_charges": 100,
        "payment_method": "bank transfer",
        "support_ticket_count": 1,
        "avg_call_minutes": 250,
        "avg_data_usage_gb": 25
    }

    # -----------------------------
    # Sample Customer 3
    # -----------------------------
    sample_customer_3 = {
        "tenure_months": 48,
        "contract_type": "two year",
        "monthly_charges": 50,
        "payment_method": "credit card",
        "support_ticket_count": 0,
        "avg_call_minutes": 270,
        "avg_data_usage_gb": 27
    }
    
    # Test all customers
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
        sample_customer_3,
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
    