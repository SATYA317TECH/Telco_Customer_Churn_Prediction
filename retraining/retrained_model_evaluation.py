import joblib

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.data_ingestion import load_churn_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from retraining.model_retraining import MODEL_PATH, load_retrained_model

TARGET = "churn"

def evaluate_retrained_model():
    artifact = load_retrained_model()

    model = artifact["model"]
    features = artifact["features"]
    threshold = artifact["threshold"]

    df = engineer_features(clean_data(load_churn_data()))

    X = df[features]
    y = df[TARGET]

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("Classification Report (Deployment Model):\n")
    print(classification_report(y, y_pred))

    print(f"\nDeployment Model Metrics:\n")
    print(f"ROC_AUC: {roc_auc_score(y, y_prob):.4f}")
    print(f"Precision: {precision_score(y, y_pred):.4f}")
    print(f"Recall: {recall_score(y, y_pred):.4f}")
    print(f"F1_Score: {f1_score(y, y_pred):.4f}")


if __name__ == "__main__":
    evaluate_retrained_model()
