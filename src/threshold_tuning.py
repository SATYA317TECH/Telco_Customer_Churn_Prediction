import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

from src.model_training import load_all_models, MODELS_ARTIFACT_PATH


def threshold_tuning():
    """
    Calculate metrics at different thresholds for Logistic Regression
    and save results to CSV for manual analysis.
    """
    print("\n" + "="*60)
    print("THRESHOLD TUNING FOR LOGISTIC REGRESSION")
    print("="*60)
    
    # Check if saved models exist
    if not os.path.exists(MODELS_ARTIFACT_PATH):
        print(f"\n❌ No saved models found at {MODELS_ARTIFACT_PATH}")
        print("   Please run src.model_training first to train and save models.")
        return None
    
    # Load saved models
    artifact = load_all_models()
    
    if artifact is None:
        return None
    
    # Extract data from artifact
    models = artifact['models']
    X_test = artifact['X_test']
    y_test = artifact['y_test']
    
    print(f"\n✅ Models loaded successfully!")
    print(f"   Test set size: {X_test.shape[0]} samples")
    
    # Check if Logistic Regression exists
    if "LogisticRegression" not in models:
        print(f"\n❌ LogisticRegression not found in saved models!")
        print(f"   Available models: {list(models.keys())}")
        return None
    
    # Get Logistic Regression model
    model = models["LogisticRegression"]
    print(f"\n🔍 Calculating thresholds for: LogisticRegression")
    
    # Get predicted probabilities for churn = 1
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Test thresholds from 0.1 to 0.9 in steps of 0.01
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    print(f"\n📊 Calculating metrics for {len(thresholds)} thresholds...")
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate confusion matrix
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))
        
        results.append({
            "threshold": round(t, 2),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = "logs/threshold_tuning_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"   File contains {len(results_df)} thresholds from 0.10 to 0.89")
    print("\n📋 First 5 rows:")
    print(results_df.head().to_string())
    print("\n📋 Last 5 rows:")
    print(results_df.tail().to_string())
    
    return results_df


if __name__ == "__main__":
    # Run threshold tuning and save to CSV
    results_df = threshold_tuning()
    
    if results_df is not None:
        print("\n" + "="*60)
        print("✅ DONE! Check logs/threshold_tuning_results.csv")
        print("   Open in Excel to find your optimal threshold")
        print("="*60)