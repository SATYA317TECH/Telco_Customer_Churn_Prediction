import joblib
from datetime import datetime
import os

from src.model_training import load_all_models, MODELS_ARTIFACT_PATH


MODEL_PATH = "artifacts/churn_model_v1.joblib"
THRESHOLD = 0.42  # Optimal threshold based on business requirements


def save_logistic_regression_model():
    """
    Save the Logistic Regression model with 0.42 threshold.
    No tuning - just loads and saves with fixed threshold.
    """
    print("\n" + "="*60)
    print("SAVING LOGISTIC REGRESSION MODEL")
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
    
    models = artifact['models']
    
    # Check if Logistic Regression exists
    if "LogisticRegression" not in models:
        print(f"\n❌ LogisticRegression not found in saved models!")
        print(f"   Available models: {list(models.keys())}")
        return None
    
    # Get Logistic Regression model
    final_model = models["LogisticRegression"]
    print(f"\n✅ Found Logistic Regression model")
    
    # Business context description
    business_context = (
        f"Balanced model with threshold={THRESHOLD:.2f}. "
        "Good trade-off between catching churners and avoiding unnecessary offers. "
        "Suitable for general retention campaigns."
    )
    
    # Create artifact
    model_artifact = {
        "model": final_model,
        "threshold": THRESHOLD,
        "model_name": "LogisticRegression",
        "threshold_source": "manual_selection",
        "business_metric_optimized": "balanced",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": business_context,
        "features": artifact.get('feature_names', [])
    }
    
    # Save model
    joblib.dump(model_artifact, MODEL_PATH)
    
    # Print success message
    print("\n" + "="*60)
    print("✅ MODEL SAVED SUCCESSFULLY")
    print("="*60)
    print(f"\n📁 Location: {MODEL_PATH}")
    print(f"🤖 Model: LogisticRegression")
    print(f"🎯 Threshold: {THRESHOLD:.2f}")
    print(f"\n📝 Business context:")
    print(f"   {business_context}")
    
    # Show file size
    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n💾 File size: {file_size:.2f} MB")

    return model_artifact


if __name__ == "__main__":
    # Save Logistic Regression model with 0.42 threshold
    artifact = save_logistic_regression_model()
    if artifact is not None:
        print("\n✅ Model artifact created and saved successfully!")
    else:
        print("\n❌ Model artifact creation failed.")