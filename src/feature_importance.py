import pandas as pd
import numpy as np

from src.model_training import load_all_models, MODELS_ARTIFACT_PATH

def logistic_feature_importance():
    # Load all models
    artifact = load_all_models()
    if artifact is None:
        return None
    models = artifact['models']

    # Get logistic regression pipeline
    log_model = models["LogisticRegression"]

    # Extract preprocessor and model
    preprocessor = log_model.named_steps["preprocessor"]
    clf = log_model.named_steps["model"]

    # Get feature names after encoding
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]

    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    all_features = np.concatenate([num_features, cat_feature_names])

    # Get coefficients
    importance = pd.DataFrame({
        "feature": all_features,
        "coefficient": clf.coef_[0]
    })

    # Absolute importance for ranking
    importance["abs_importance"] = importance["coefficient"].abs()

    return importance.sort_values(
        by="abs_importance",
        ascending=False
    )

if __name__ == "__main__":
    fi = logistic_feature_importance()
    print(fi)