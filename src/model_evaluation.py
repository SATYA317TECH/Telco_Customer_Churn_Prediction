from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score
)
import pandas as pd
import os

from src.model_training import load_all_models, MODELS_ARTIFACT_PATH

def evaluate_models():
    """
    Evaluate all saved models on test set and return metrics.
    Loads models from artifacts instead of retraining.
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Check if saved models exist
    if not os.path.exists(MODELS_ARTIFACT_PATH):
        print(f"\n❌ No saved models found at {MODELS_ARTIFACT_PATH}")
        print("   Please run src.model_training first to train and save models.")
        return None
    
    # Load saved models and test data
    artifact = load_all_models()
    
    if artifact is None:
        return None
    
    # Extract data from artifact
    models = artifact['models']
    X_test = artifact['X_test']
    y_test = artifact['y_test']
    training_date = artifact['training_date']
    
    print(f"\n✅ Models loaded successfully!")
    print(f"   Training date: {training_date}")
    print(f"   Test set size: {X_test.shape[0]} samples")
    print(f"   Number of models: {len(models)}")
    
    results_list = []
    classification_reports_text = []

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "ROC_AUC": round(roc_auc_score(y_test, y_prob),4),
            "Accuracy": round(accuracy_score(y_test, y_pred),4),
            "Precision": round(precision_score(y_test, y_pred),4),
            "Recall": round(recall_score(y_test, y_pred),4),
            "F1_Score": round(f1_score(y_test, y_pred),4)
        }

        results_list.append(metrics)

        # ------------------------
        # CLASSIFICATION REPORT TEXT
        # ------------------------
        report = classification_report(y_test, y_pred)

        classification_reports_text.append(
            f"\n{'='*60}\n"
            f"MODEL: {name}\n"
            f"{'='*60}\n"
            f"{report}\n"
        )
    # ==========================
    # SAVE METRICS CSV
    # ==========================
    print("\n📊 Compiling model comparison results...")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Sort by ROC_AUC (best first)
    results_df = results_df.sort_values(by="ROC_AUC", ascending=False)

    # Save to CSV
    metrics_path = os.path.join("logs", "model_comparison_results.csv")
    results_df.to_csv(metrics_path, index=False)

    print(f"📊 Model comparison saved to: {metrics_path}")

    # ==========================
    # SAVE CLASSIFICATION REPORT TXT
    # ==========================
    print("\n📊 Saving classification reports for all models...")

    report_path = os.path.join("logs", "all_classification_reports.txt")

    with open(report_path, "w") as f:
        f.writelines(classification_reports_text)

    print(f"✅ Classification reports saved to: {report_path}")

    return results_df


if __name__ == "__main__":
    results = evaluate_models()
    if results is not None:
        print("\n✅ Model evaluation completed successfully!")