from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import pandas as pd
import joblib
import os
from datetime import datetime
warnings.filterwarnings('ignore')

from src.data_ingestion import get_connection  # Import connection function

MODEL_PATH = "artifacts/churn_deployment_model.joblib"

# Only the 7 selected important features
selected_features = [
    "tenure_months",
    "contract_type",
    "monthly_charges",
    "payment_method",
    "support_ticket_count",
    "avg_call_minutes",
    "avg_data_usage_gb"
]

def load_deployment_data():
    """
    Load data directly from the deployment view in SQL.
    This ensures we're using the exact same features as production.
    """
    print("\n📥 Loading data from SQL deployment view...")
    
    query = "SELECT * FROM vw_churn_deployment_features"
    
    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
        
        print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"   Features: {list(df.columns)}")
        
        # Check churn distribution
        churn_dist = df['churn'].value_counts(normalize=True)
        print(f"   Churn rate: {churn_dist.get(1, 0)*100:.2f}%")
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def prepare_retrained_dataset():
    """
    Prepare dataset for retraining using SQL view.
    No data cleaning needed as view already has clean data.
    """
    # Load data directly from SQL view
    df = load_deployment_data()
    
    if df is None:
        raise Exception("Failed to load data from SQL view")
    
    # Separate features and target
    X = df[selected_features]
    y = df["churn"]
    
    print(f"\n📊 Training data shape: {X.shape}")
    print(f"   Features: {len(selected_features)}")
    print(f"   Target distribution:\n{y.value_counts(normalize=True)}")
    
    # Define preprocessing
    numerical_features = [
        "tenure_months",
        "monthly_charges",
        "support_ticket_count",
        "avg_call_minutes",
        "avg_data_usage_gb"
    ]
    
    categorical_features = [
        "contract_type",
        "payment_method"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    
    return X, y, preprocessor


def cross_validate_model(pipeline, X, y):
    print("\n" + "="*50)
    print("Running 5-Fold Cross Validation on Retrained Model...")
    print("="*50)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    
    print(f"\n📊 ROC-AUC Scores: {[round(s, 4) for s in scores]}")
    print(f"📈 Mean ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    
    return scores.mean()


def tune_logistic_regression(base_pipeline, X, y):
    print("\n" + "="*50)
    print("Tuning Retrained Logistic Regression using GridSearchCV...")
    print("="*50)
    
    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 50],
        "model__solver": ["liblinear", "lbfgs"],
        "model__penalty": ["l2"],
        "model__class_weight": ["balanced", None]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    
    print("\n🔄 Searching over parameter grid...")
    grid.fit(X, y)
    
    print(f"\n✅ Best Parameters: {grid.best_params_}")
    print(f"🏆 Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


def retrain_model():
    """
    Retrain model using data from SQL deployment view.
    """
    print("\n" + "="*60)
    print("RETRAINING DEPLOYMENT MODEL (7 FEATURES)")
    print("="*60)
    
    # Load and prepare data from SQL view
    X, y, preprocessor = prepare_retrained_dataset()
    
    # Create base pipeline
    base_pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
    
    # 1. Cross Validation
    cv_score = cross_validate_model(base_pipeline, X, y)
    
    # 2. Grid Search for Best Parameters
    best_model = tune_logistic_regression(base_pipeline, X, y)
    
    # 3. Final Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print("\n" + "="*50)
    print("Training final retrained model...")
    print("="*50)
    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    best_model.fit(X_train, y_train)
    
    # Quick evaluation on test set
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    test_auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\n📈 Test ROC-AUC: {test_auc:.4f}")
    
    # 4. Save Final Model
    artifact = {
        "model": best_model,
        "threshold": 0.42,  # Updated to match your chosen threshold
        "model_name": "Retrained LogisticRegression (7 Features)",
        "features": selected_features,
        "cv_score": round(cv_score, 4),
        "test_auc": round(test_auc, 4),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Deployment model using only 7 features from SQL view",
        "data_source": "vw_churn_deployment_features"
    }
    
    joblib.dump(artifact, MODEL_PATH)
    
    print("\n" + "="*60)
    print("✅ RETRAINED MODEL SAVED SUCCESSFULLY")
    print("="*60)
    print(f"   Location: {MODEL_PATH}")
    print(f"   Features: {len(selected_features)}")
    print(f"   Threshold: 0.42")
    print(f"   CV ROC-AUC: {cv_score:.4f}")
    print(f"   Test ROC-AUC: {test_auc:.4f}")

def load_retrained_model():
    """
    Load the retrained model from the saved artifact.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No model found at {MODEL_PATH}")
        print("   Please run retrain_model() first.")
        return None
    
    print(f"\n📂 Loading retrained model from {MODEL_PATH}...")
    artifact = joblib.load(MODEL_PATH)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Training date: {artifact['created_at']}")
    print(f"   Model name: {artifact['model_name']}")
    
    return artifact

if __name__ == "__main__":
    retrain_model()