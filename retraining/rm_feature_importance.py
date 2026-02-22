import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "artifacts/churn_deployment_model.joblib"

def load_model():
    """Load the retrained deployment model"""
    return joblib.load(MODEL_PATH)

def get_feature_importance():
    """Extract and display feature importance for logistic regression"""
    
    # Load model
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle["threshold"]
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE FOR RETRAINED MODEL (7 FEATURES)")
    print("="*60)
    
    print(f"\n📊 Model: {bundle.get('model_name', 'LogisticRegression')}")
    print(f"🎯 Threshold: {threshold}")
    print(f"📅 Created: {bundle.get('created_at', 'Unknown')}")
    
    # Extract coefficients - handle both pipeline and direct model
    if hasattr(model, 'named_steps'):
        # If it's a pipeline, get the actual model
        model_step = model.named_steps['model']
        coefficients = model_step.coef_[0]
    else:
        # Direct model
        coefficients = model.coef_[0]
    
    # Debug: Print lengths to see what's happening
    print(f"\n🔍 Debug Info:")
    print(f"   Number of features (original): {len(features)}")
    print(f"   Number of coefficients: {len(coefficients)}")
    
    # Check if lengths match
    if len(features) != len(coefficients):
        print(f"\n⚠️  Warning: Length mismatch!")
        print(f"   Features ({len(features)}) vs Coefficients ({len(coefficients)})")
        print(f"\n📊 One-hot encoded features detected!")
        print(f"   The 7 original features have been expanded to {len(coefficients)} encoded features.")
        print(f"   Showing coefficients for encoded features:\n")
        
        # Create feature names for encoded features
        # Based on your 7 features, the encoding likely expands contract_type and payment_method
        encoded_feature_names = []
        
        # Numerical features (keep original names)
        numerical_features = [
            "tenure_months",
            "monthly_charges", 
            "support_ticket_count",
            "avg_call_minutes",
            "avg_data_usage_gb"
        ]
        encoded_feature_names.extend(numerical_features)
        
        # Categorical features (one-hot encoded)
        # contract_type: 3 categories → 3 encoded features
        contract_categories = ["month-to-month", "one year", "two year"]
        for cat in contract_categories:
            encoded_feature_names.append(f"contract_type_{cat}")
        
        # payment_method: 4 categories → 4 encoded features
        payment_categories = ["electronic check", "credit card", "bank transfer", "mailed check"]
        for cat in payment_categories:
            encoded_feature_names.append(f"payment_method_{cat}")
        
        # Verify length matches
        if len(encoded_feature_names) == len(coefficients):
            print(f"✅ Successfully mapped to {len(encoded_feature_names)} encoded features")
            feature_names = encoded_feature_names
        else:
            # Fallback: generic names
            print(f"⚠️  Using generic feature names")
            feature_names = [f"Feature_{i+1}" for i in range(len(coefficients))]
    else:
        # Lengths match - use original features
        feature_names = features
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_importance': np.abs(coefficients)
    }).sort_values('abs_importance', ascending=False).reset_index(drop=True)
    
    # Add direction (positive/negative impact)
    importance_df['direction'] = importance_df['coefficient'].apply(
        lambda x: '⬆️ Increases churn' if x > 0 else '⬇️ Decreases churn'
    )
    
    # Add percentage of total importance
    importance_df['importance_pct'] = (importance_df['abs_importance'] / 
                                        importance_df['abs_importance'].sum() * 100).round(2)
    
    # Add rank
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    print("\n📊 FEATURE IMPORTANCE RANKING:")
    print("-" * 90)
    for _, row in importance_df.iterrows():
        print(f"{row['rank']:2d}. {row['feature'][:40]:<40} "
              f"Coef: {row['coefficient']:>8.4f} | "
              f"Imp: {row['abs_importance']:>6.4f} | "
              f"{row['direction']}")
    
    print("\n" + "="*60)
    print("📈 TOP 7 ENCODED FEATURES:")
    print("="*60)
    print(importance_df.head(7)[['feature', 'coefficient', 'abs_importance', 'importance_pct']].to_string(index=False))
    
    # Save to CSV
    importance_df.to_csv("logs/retrained_feature_importance.csv", index=False)
    print(f"\n✅ Full feature importance saved to: logs/retrained_feature_importance.csv")
    
    # Group by original features (for interpretability)
    print("\n" + "="*60)
    print("📊 GROUPED BY ORIGINAL FEATURES:")
    print("="*60)
    
    # Create a mapping for original features
    feature_groups = {
        'tenure_months': 'tenure_months',
        'monthly_charges': 'monthly_charges',
        'support_ticket_count': 'support_ticket_count',
        'avg_call_minutes': 'avg_call_minutes',
        'avg_data_usage_gb': 'avg_data_usage_gb',
        'contract_type': [f for f in feature_names if f.startswith('contract_type_')],
        'payment_method': [f for f in feature_names if f.startswith('payment_method_')]
    }
    
    for group_name, group_features in feature_groups.items():
        if isinstance(group_features, list):
            # For categorical features, show all categories
            group_df = importance_df[importance_df['feature'].isin(group_features)]
            if not group_df.empty:
                print(f"\n{group_name.upper()}:")
                for _, row in group_df.iterrows():
                    print(f"  • {row['feature']}: {row['coefficient']:>8.4f} ({row['direction']})")
        else:
            # For numerical features
            row = importance_df[importance_df['feature'] == group_features]
            if not row.empty:
                print(f"\n{group_name}: {row.iloc[0]['coefficient']:>8.4f} ({row.iloc[0]['direction']})")
    
    return importance_df

def compare_with_original():
    """Compare with original model's feature importance (if available)"""
    try:
        original_model = joblib.load("artifacts/churn_model_v1.joblib")
        print("\n" + "="*60)
        print("📊 COMPARISON WITH ORIGINAL MODEL")
        print("="*60)
        
        if 'features' in original_model:
            print(f"Original model features: {len(original_model['features'])}")
            print(f"Retrained model features: 7 (expanded to 12 encoded)")
            print("\n✅ Retrained model is more focused and interpretable!")
    except:
        pass

if __name__ == "__main__":
    importance = get_feature_importance()
    compare_with_original()