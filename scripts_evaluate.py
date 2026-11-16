import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, classification_report
)
from src.inference import predict_traffic

BASE = r"D:\IDS research\outputs"
OUT = r"D:\ARGUS\artifacts\evaluation"
os.makedirs(OUT, exist_ok=True)

tests = [
    ("Binary", "binary", "lr", "X_test.csv", "y_test.csv"),
    ("Binary", "binary", "rf", "X_test.csv", "y_test.csv"),
    ("Multiclass", "multiclass", "lr", "X_test_multi_class.csv", "y_test_multi_class.csv"),
    ("Multiclass", "multiclass", "rf", "X_test_multi_class.csv", "y_test_multi_class.csv"),
]

summary = []

for name, task, model, xf, yf in tests:
    print(f"\nStarting {name} {model.upper()} evaluation...", flush=True)

    x_iter = pd.read_csv(os.path.join(BASE, xf), chunksize=10000)
    y_iter = pd.read_csv(os.path.join(BASE, yf), chunksize=10000)

    actual_all, predicted_all, score_all = [], [], []
    start = time.perf_counter()
    processed = 0

    for X, Y in zip(x_iter, y_iter):
        result = predict_traffic(X, task=task, model_type=model)

        actual = Y.iloc[:, 0].to_numpy(dtype=int)
        predicted = result["Predicted Class ID"].to_numpy(dtype=int)

        actual_all.extend(actual)
        predicted_all.extend(predicted)

        if task == "binary" and "Attack Score" in result.columns:
            score_all.extend(result["Attack Score"].to_numpy(dtype=float))

        processed += len(X)
        if processed % 50000 == 0:
            print(f"  Processed {processed:,} rows", flush=True)

    elapsed = time.perf_counter() - start
    y_true = np.asarray(actual_all)
    y_pred = np.asarray(predicted_all)

    metrics = {
        "Task": name,
        "Model": model.upper(),
        "Rows": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Prediction_Time_Seconds": elapsed,
    }

    if task == "binary":
        metrics.update({
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        })

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp,
            "Specificity": tn / (tn + fp) if tn + fp else 0,
            "FPR": fp / (fp + tn) if fp + tn else 0,
            "FNR": fn / (fn + tp) if fn + tp else 0,
        })

        if len(score_all) == len(y_true):
            metrics["PR_AUC"] = average_precision_score(y_true, score_all)

        pd.DataFrame(cm, index=["Actual_0", "Actual_1"],
                     columns=["Predicted_0", "Predicted_1"]).to_csv(
            os.path.join(OUT, f"binary_{model}_confusion_matrix.csv")
        )

    else:
        metrics.update({
            "Weighted_Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Weighted_Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        })

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(
            os.path.join(OUT, f"multiclass_{model}_classification_report.csv")
        )

        cm = confusion_matrix(y_true, y_pred)
        pd.DataFrame(cm).to_csv(
            os.path.join(OUT, f"multiclass_{model}_confusion_matrix.csv"),
            index=False
        )

    summary.append(metrics)
    print(f"Completed {name} {model.upper()} — {len(y_true):,} rows in {elapsed:.1f}s", flush=True)

pd.DataFrame(summary).to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)

print("\nALL EVALUATIONS COMPLETED")
print(pd.DataFrame(summary).to_string(index=False))
print(f"\nSaved results to: {OUT}")
