from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import warnings
import joblib
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from src.encoding_scaling import encode_and_scale

# Path to save all models
MODELS_ARTIFACT_PATH = "artifacts/all_trained_models.joblib"


def evaluate_with_cross_validation(name, pipeline, X, y):
    """
    Perform 5-fold cross validation and return mean ROC-AUC score.
    """
    print(f"\n{'='*50}")
    print(f"Running 5-Fold Cross Validation for {name}...")
    print(f"{'='*50}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    print(f"\n📊 ROC-AUC Scores: {[round(s, 4) for s in scores]}")
    print(f"📈 Mean ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return scores.mean()


def train_and_save_all_models():
    """
    Train multiple models and save ALL trained models to a single file.
    No tuning, no selection - just training and saving.
    """
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE - TRAINING ALL MODELS")
    print("="*60)
    
    # Load and prepare data
    print("\n📥 Loading and preparing features...")
    X, y, preprocessor, feature_names = encode_and_scale()
    
    print(f"\n📊 Dataset shape: {X.shape}")
    print(f"🎯 Target distribution:\n{y.value_counts(normalize=True).mul(100).round(2)}")
    
    # Calculate class weight for XGBoost
    scale_pos_weight = y.value_counts()[0] / y.value_counts()[1]
    print(f"\n⚖️  Scale pos weight for XGBoost: {scale_pos_weight:.2f}")
    
    # ============================================================================
    # DEFINE ALL MODELS WITH DEFAULT PARAMETERS (NO TUNING)
    # ============================================================================
    print("\n🔧 Defining models with default parameters...")
    
    models = {
        "LogisticRegression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        "RandomForest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=100,  # Default
                max_depth=None,     # Default
                min_samples_split=2, # Default
                min_samples_leaf=1,  # Default
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        "GradientBoosting": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingClassifier(
                n_estimators=100,  # Default
                learning_rate=0.1,  # Default
                max_depth=3,        # Default
                min_samples_split=2, # Default
                min_samples_leaf=1,  # Default
                subsample=1.0,       # Default
                random_state=42
            ))
        ]),
        
        "XGBoost": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=100,  # Default
                learning_rate=0.3,  # Default
                max_depth=6,        # Default
                min_child_weight=1,  # Default
                subsample=1.0,       # Default
                colsample_bytree=1.0, # Default
                gamma=0,             # Default
                reg_alpha=0,         # Default
                reg_lambda=1,        # Default
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        
        "LightGBM": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(
                n_estimators=100,  # Default
                learning_rate=0.1,  # Default
                num_leaves=31,      # Default
                max_depth=-1,        # Default
                min_child_samples=20, # Default
                subsample=1.0,       # Default
                colsample_bytree=1.0, # Default
                reg_alpha=0.0,       # Default
                reg_lambda=0.0,      # Default
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])
    }
    
    # ============================================================================
    # CROSS VALIDATION (Optional - can be commented out if you just want training)
    # ============================================================================
    print("\n" + "="*60)
    print("CROSS VALIDATION RESULTS")
    print("="*60)
    
    cv_results = {}
    
    for name, pipeline in models.items():
        score = evaluate_with_cross_validation(name, pipeline, X, y)
        cv_results[name] = score
    
    # Display CV summary
    print("\n" + "="*50)
    print("📊 CROSS VALIDATION SUMMARY")
    print("="*50)
    
    cv_df = pd.DataFrame({
        'Model': list(cv_results.keys()),
        'ROC-AUC': list(cv_results.values())
    }).sort_values('ROC-AUC', ascending=False)
    
    print("\n" + cv_df.to_string(index=False))
    print(f"\n🏆 Best model from CV: {cv_df.iloc[0]['Model']} with ROC-AUC = {cv_df.iloc[0]['ROC-AUC']:.4f}")
    
    # ============================================================================
    # FINAL TRAIN-TEST SPLIT
    # ============================================================================
    print("\n" + "="*60)
    print("FINAL TRAINING ON TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    print(f"🎯 Train churn rate: {y_train.mean()*100:.2f}%")
    print(f"🎯 Test churn rate: {y_test.mean()*100:.2f}%")
    
    # Train all models on train-test split
    trained_models = {}
    training_times = {}
    test_metrics = {}
    
    import time
    
    for name, pipeline in models.items():
        print(f"\n🔄 Training {name}...")
        start_time = time.time()
        
        pipeline.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        training_times[name] = elapsed_time
        
        # Quick evaluation on test set
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        test_auc = roc_auc_score(y_test, y_pred_prob)
        
        # Store test metrics
        test_metrics[name] = {
            'roc_auc': test_auc,
            'training_time': elapsed_time
        }
        
        print(f"   ✅ Training time: {elapsed_time:.2f} seconds")
        print(f"   📈 Test ROC-AUC: {test_auc:.4f}")
        
        trained_models[name] = pipeline
    
    # ============================================================================
    # SAVE ALL MODELS TO A SINGLE FILE
    # ============================================================================
    print("\n" + "="*60)
    print("💾 SAVING ALL MODELS")
    print("="*60)
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(os.path.dirname(MODELS_ARTIFACT_PATH), exist_ok=True)
    
    # Prepare artifact to save
    artifact = {
        "models": trained_models,
        "feature_names": feature_names,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "training_times": training_times,
        "X_test": X_test,  # Save test data for later evaluation
        "y_test": y_test,  # Save test labels for later evaluation
        "preprocessor": preprocessor,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "All trained models with default parameters"
    }
    
    # Save to file
    joblib.dump(artifact, MODELS_ARTIFACT_PATH)
    print(f"\n✅ All models saved successfully to: {MODELS_ARTIFACT_PATH}")
    print(f"   File size: {os.path.getsize(MODELS_ARTIFACT_PATH) / 1024 / 1024:.2f} MB")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    
    print("\n📋 Models trained and saved:")
    for name in trained_models.keys():
        print(f"  • {name}")
    
    print(f"\n⏱️  Total training time: {sum(training_times.values()):.2f} seconds")
    print(f"\n📁 Models saved at: {MODELS_ARTIFACT_PATH}")
    
    return trained_models, X_test, y_test, feature_names


def load_all_models():
    """
    Load all previously trained models from the saved artifact.
    """
    if not os.path.exists(MODELS_ARTIFACT_PATH):
        print(f"❌ No saved models found at {MODELS_ARTIFACT_PATH}")
        print("   Please run train_and_save_all_models() first.")
        return None
    
    print(f"\n📂 Loading models from {MODELS_ARTIFACT_PATH}...")
    artifact = joblib.load(MODELS_ARTIFACT_PATH)
    
    print(f"✅ Models loaded successfully!")
    print(f"   Training date: {artifact['training_date']}")
    print(f"   Models available: {list(artifact['models'].keys())}")
    
    return artifact


if __name__ == "__main__":
    # Train and save all models
    models, X_test, y_test, feature_names = train_and_save_all_models()
    
    # Show feature names for reference
    print("\n" + "="*60)
    print("FEATURE NAMES")
    print("="*60)
    for i, name in enumerate(feature_names[:20]):
        print(f"{i+1:2d}. {name}")
    
    print("\n✅ Model training and saving completed successfully!")
