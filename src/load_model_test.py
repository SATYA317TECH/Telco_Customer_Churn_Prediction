import joblib

MODEL_PATH = "artifacts/churn_model_v1.joblib"

def load_and_test_model():
    """
    Load the saved model and display all available information.
    """
    print("\n" + "="*60)
    print("LOADING SAVED CHURN MODEL")
    print("="*60)
    
    try:
        # Load the artifact
        artifact = joblib.load(MODEL_PATH)
        
        # Extract components
        model = artifact["model"]
        threshold = artifact["threshold"]
        model_name = artifact.get("model_name", "Unknown")
        created_at = artifact.get("created_at", "Unknown")
        description = artifact.get("description", "No description")
        threshold_source = artifact.get("threshold_source", "Unknown")
        business_metric = artifact.get("business_metric_optimized", "Unknown")
        features = artifact.get("features", None)
        
        # Print model information
        print(f"\n📁 Model file: {MODEL_PATH}")
        print(f"\n🤖 Model Information:")
        print(f"   • Name: {model_name}")
        print(f"   • Type: {type(model.named_steps['model'] if hasattr(model, 'named_steps') else model).__name__}")
        print(f"   • Created: {created_at}")
        
        print(f"\n🎯 Threshold Information:")
        print(f"   • Current threshold: {threshold:.2f}")
        print(f"   • Threshold source: {threshold_source}")
        print(f"   • Optimized for: {business_metric}")
        
        print(f"\n📝 Description:")
        print(f"   {description}")
        
        # Feature information
        if features is not None:
            print(f"\n🔢 Feature Information:")
            print(f"   • Total features: {len(features)}")
            print(f"   • First 10 features: {features[:10]}")
        
        # Model parameters
        print(f"\n⚙️ Model Parameters:")
        if hasattr(model, 'named_steps'):
            model_step = model.named_steps['model']
            if hasattr(model_step, 'get_params'):
                params = model_step.get_params()
                # Show only important parameters
                important_params = {}
                for key in ['C', 'max_iter', 'n_estimators', 'max_depth', 
                           'learning_rate', 'class_weight']:
                    if key in params:
                        important_params[key] = params[key]
                
                if important_params:
                    for key, value in important_params.items():
                        print(f"   • {key}: {value}")
                
        return artifact
        
    except FileNotFoundError:
        print(f"\n❌ Error: Model file not found at {MODEL_PATH}")
        print("   Please run src.save_model first to create the model.")
        return None
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Load and test the saved model
    artifact = load_and_test_model()
    
    if artifact is not None:
        print("\n")
        print("✅ Model loaded and tested successfully!")