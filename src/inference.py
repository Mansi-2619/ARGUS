from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"
PREPROCESS_DIR = ROOT / "artifacts" / "preprocessing"


def load_artifacts(task="binary", model_type="lr"):
    """Load the selected model and its matching preprocessing artifacts."""

    if task not in ("binary", "multiclass"):
        raise ValueError("Task must be 'binary' or 'multiclass'.")

    if model_type not in ("lr", "rf"):
        raise ValueError("Model type must be 'lr' or 'rf'.")

    suffix = "_multiclass" if task == "multiclass" else ""
    model_name = "logistic_regression" if model_type == "lr" else "random_forest"

    model_path = MODEL_DIR / f"{model_name}{suffix}.pkl"
    scaler_path = PREPROCESS_DIR / f"robust_scaler_{task}.pkl"
    log_path = PREPROCESS_DIR / "log_features.pkl"

    required = [model_path, scaler_path, log_path]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required ARGUS artifacts:\n" + "\n".join(missing)
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    log_features = joblib.load(log_path)

    if isinstance(log_features, dict):
        log_features = list(log_features.keys())
    else:
        log_features = list(log_features)

    encoder = None
    if task == "multiclass":
        encoder_path = PREPROCESS_DIR / "label_encoder_multiclass.pkl"
        if encoder_path.exists():
            encoder = joblib.load(encoder_path)

    return model, scaler, log_features, encoder


def predict_traffic(data, task="binary", model_type="lr"):
    """
    Predict traffic using Logistic Regression or Random Forest.

    Input must contain the original 78 CICIDS2017 features.
    Extra columns are ignored.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("The uploaded CSV contains no traffic records.")

    model, scaler, log_features, encoder = load_artifacts(task, model_type)

    feature_cols = list(model.feature_names_in_)

    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        preview = ", ".join(missing_cols[:10])
        raise ValueError(
            f"CSV is missing {len(missing_cols)} required features: {preview}"
        )

    X = data.loc[:, feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.astype("float64")

    invalid = X.isna() | ~np.isfinite(X)
    if invalid.any().any():
        bad_cols = invalid.sum()
        bad_cols = bad_cols[bad_cols > 0].to_dict()
        details = ", ".join(
            f"{col}: {count}" for col, count in list(bad_cols.items())[:10]
        )
        raise ValueError(
            "Invalid or missing numeric values found. "
            f"Please clean the CSV before prediction. Details: {details}"
        )

    # Preserve the verified research log transformation.
    unknown_log_cols = [c for c in log_features if c not in feature_cols]
    if unknown_log_cols:
        raise ValueError("The saved log-feature list does not match the model.")

    X.loc[:, log_features] = np.log1p(
        X.loc[:, log_features].clip(lower=0)
    )

    # Use the saved task-specific scaler. Never fit a new scaler.
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    predictions = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)
        confidence = probabilities.max(axis=1)
    else:
        probabilities = None
        confidence = np.full(len(predictions), np.nan)

    results = pd.DataFrame({
        "Row": np.arange(1, len(data) + 1),
        "Predicted Class ID": predictions,
        "Prediction": predictions.astype(str),
        "Model Score": confidence
    })

    if task == "binary":
        results["Prediction"] = np.where(
            predictions == 1, "Attack", "Benign"
        )

        if probabilities is not None:
            attack_indices = np.where(model.classes_ == 1)[0]
            if len(attack_indices):
                results["Attack Score"] = probabilities[:, attack_indices[0]]

    else:
        if encoder is not None:
            results["Prediction"] = encoder.inverse_transform(
                np.asarray(predictions, dtype=int)
            )
        else:
            results["Prediction"] = [
                f"Class {value}" for value in predictions
            ]

    model_label = "Logistic Regression" if model_type == "lr" else "Random Forest"
    task_label = "Binary" if task == "binary" else "Multiclass"
    results["Model"] = f"{model_label} - {task_label}"

    return results

