
import numpy as np
import pandas as pd
import shap

from src.inference import load_artifacts


def _prepare_explanation_data(data, task):
    """Apply the exact saved research preprocessing for the selected model."""

    if data.empty:
        raise ValueError("No traffic records available.")

    model, scaler, log_features, label_encoder = load_artifacts(task)

    feature_cols = list(model.feature_names_in_)

    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing[:10]}")

    X = data.loc[:, feature_cols].copy()

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.astype("float64")

    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Traffic data contains missing or invalid values.")

    # Preserve the verified research log transformation.
    X.loc[:, log_features] = np.log1p(
        X.loc[:, log_features].clip(lower=0)
    )

    # Use the saved scaler. Never fit a scaler during explanation.
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols
    )

    return model, X_scaled, feature_cols, label_encoder


def _get_contributions(model, X_scaled, row_index, background_size, class_index=None):
    """Calculate SHAP contributions for one selected record."""

    background = X_scaled.head(min(background_size, len(X_scaled)))

    explainer = shap.LinearExplainer(model, background)
    explanation = explainer(X_scaled.iloc[[row_index]])

    values = np.asarray(explanation.values)

    if values.ndim == 3:
        if class_index is None:
            raise ValueError("A class index is required for multiclass SHAP.")

        if values.shape[-1] == len(model.classes_):
            contributions = values[0, :, class_index]
        elif values.shape[1] == len(model.classes_):
            contributions = values[0, class_index, :]
        else:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

    elif values.ndim == 2:
        contributions = values[0]

    else:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    return contributions


def _build_result_table(feature_cols, contributions, positive_direction, negative_direction):
    """Build a consistently structured feature contribution table."""

    results = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP Value": contributions
    })

    results["Direction"] = np.where(
        results["SHAP Value"] > 0,
        positive_direction,
        negative_direction
    )

    results["Absolute Impact"] = results["SHAP Value"].abs()

    return results.sort_values(
        "Absolute Impact",
        ascending=False
    ).reset_index(drop=True)


def explain_record(data, row_index=0, background_size=50):
    """Explain one binary Logistic Regression prediction using SHAP."""

    if row_index < 0 or row_index >= len(data):
        raise IndexError("Selected record does not exist.")

    model, X_scaled, feature_cols, _ = _prepare_explanation_data(
        data, "binary"
    )

    class_index = list(model.classes_).index(1)

    contributions = _get_contributions(
        model,
        X_scaled,
        row_index,
        background_size,
        class_index
    )

    return _build_result_table(
        feature_cols,
        contributions,
        "Pushes towards Attack",
        "Pushes towards Benign"
    )


def explain_multiclass_record(data, row_index=0, background_size=50):
    """Explain the predicted class of one multiclass Logistic Regression record."""

    if row_index < 0 or row_index >= len(data):
        raise IndexError("Selected record does not exist.")

    model, X_scaled, feature_cols, label_encoder = _prepare_explanation_data(
        data, "multiclass"
    )

    # Identify the class predicted for this particular record.
    predicted_encoded = model.predict(
        X_scaled.iloc[[row_index]]
    )[0]

    class_index = list(model.classes_).index(predicted_encoded)

    if label_encoder is not None:
        predicted_class = str(
            label_encoder.inverse_transform(
                np.asarray([predicted_encoded], dtype=int)
            )[0]
        )
    else:
        predicted_class = str(predicted_encoded)

    contributions = _get_contributions(
        model,
        X_scaled,
        row_index,
        background_size,
        class_index
    )

    results = _build_result_table(
        feature_cols,
        contributions,
        f"Pushes towards {predicted_class}",
        f"Pushes away from {predicted_class}"
    )

    results.attrs["predicted_class"] = predicted_class

    return results
