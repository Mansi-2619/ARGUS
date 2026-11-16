import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join("artifacts", "evaluation")

st.set_page_config(page_title="ARGUS | Model Comparison", page_icon="📊", layout="wide")

st.title("Model Performance Comparison")
st.caption("Evaluation of Logistic Regression and Random Forest on 756,031 held-out records.")

metrics_path = os.path.join(BASE, "model_comparison.csv")

if not os.path.exists(metrics_path):
    st.error("Evaluation results not found. Run scripts_evaluate.py first.")
    st.stop()

df = pd.read_csv(metrics_path)

binary = df[df["Task"] == "Binary"].set_index("Model")
multi = df[df["Task"] == "Multiclass"].set_index("Model")

st.subheader("Binary Classification")
cols = st.columns(4)
for col, label, key, fmt in zip(
    cols,
    ["Accuracy", "Precision", "Recall", "F1 Score"],
    ["Accuracy", "Precision", "Recall", "F1"],
    ["{:.2%}", "{:.2%}", "{:.2%}", "{:.2%}"]
):
    col.metric(label, fmt.format(binary.loc["RF", key]))

st.write("**Binary model comparison**")
binary_view = binary[[
    "Accuracy", "Precision", "Recall", "F1",
    "PR_AUC", "False_Negatives", "FNR", "FPR",
    "Prediction_Time_Seconds"
]].T
st.dataframe(binary_view.style.format({
    "LR": "{:.4f}", "RF": "{:.4f}"
}), use_container_width=True)

fig, ax = plt.subplots(figsize=(9, 4))
binary[["Accuracy", "Precision", "Recall", "F1"]].T.plot(
    kind="bar", ax=ax
)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Binary Classification Metrics")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Model")
st.pyplot(fig)
plt.close(fig)

st.subheader("Binary Confusion Matrices")
c1, c2 = st.columns(2)

for col, model in [(c1, "lr"), (c2, "rf")]:
    path = os.path.join(BASE, f"binary_{model}_confusion_matrix.csv")
    if os.path.exists(path):
        cm = pd.read_csv(path, index_col=0)
        with col:
            st.write(f"**{model.upper()}**")
            st.dataframe(cm, use_container_width=True)

st.divider()
st.subheader("Multiclass Classification")

cols = st.columns(4)
for col, label, key in zip(
    cols,
    ["Accuracy", "Weighted F1", "Macro F1", "Prediction time"],
    ["Accuracy", "Weighted_F1", "Macro_F1", "Prediction_Time_Seconds"]
):
    value = multi.loc["RF", key]
    col.metric(label, f"{value:.2%}" if key != "Prediction_Time_Seconds" else f"{value:.1f} sec")

st.write("**Multiclass model comparison**")
multi_view = multi[[
    "Accuracy", "Weighted_Precision", "Weighted_Recall",
    "Weighted_F1", "Macro_Precision", "Macro_Recall",
    "Macro_F1", "Prediction_Time_Seconds"
]].T
st.dataframe(multi_view.style.format({
    "LR": "{:.4f}", "RF": "{:.4f}"
}), use_container_width=True)

fig, ax = plt.subplots(figsize=(9, 4))
multi[["Accuracy", "Weighted_F1", "Macro_F1"]].T.plot(
    kind="bar", ax=ax
)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Multiclass Classification Metrics")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Model")
st.pyplot(fig)
plt.close(fig)

st.subheader("Multiclass Confusion Matrices")
for model in ["lr", "rf"]:
    path = os.path.join(BASE, f"multiclass_{model}_confusion_matrix.csv")
    if os.path.exists(path):
        cm = pd.read_csv(path, header=None).to_numpy()
        with st.expander(f"{model.upper()} — Confusion Matrix"):
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm, aspect="auto")
            ax.set_title(f"Multiclass {model.upper()} Confusion Matrix")
            ax.set_xlabel("Predicted class ID")
            ax.set_ylabel("Actual class ID")
            fig.colorbar(im, ax=ax, label="Records")
            st.pyplot(fig)
            plt.close(fig)

    report_path = os.path.join(
        BASE, f"multiclass_{model}_classification_report.csv"
    )
    if os.path.exists(report_path):
        with st.expander(f"{model.upper()} — Per-class Classification Report"):
            report = pd.read_csv(report_path, index_col=0)
            st.dataframe(report, use_container_width=True)

st.divider()
st.subheader("Research Findings")

st.markdown("""
- **Binary detection:** Random Forest achieved higher recall and substantially fewer false negatives.
- **Multiclass detection:** Random Forest achieved a much stronger macro F1, indicating better performance across classes.
- **Logistic Regression:** Provides a faster, simpler baseline and supports linear-model explainability.
- **Explainability:** ARGUS integrates SHAP explanations to show which traffic features influence individual predictions.
""")

st.caption(
    "Prediction time is measured during this evaluation on the current machine. "
    "It includes inference and preprocessing; it is not a training-time measurement."
)

with open(metrics_path, "rb") as file:
    st.download_button(
        "Download Model Comparison Metrics",
        file,
        file_name="ARGUS_model_comparison.csv",
        mime="text/csv"
    )
