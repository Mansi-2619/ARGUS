import streamlit as st
import pandas as pd

from src.inference import predict_traffic
from src.explainability import (
    explain_record,
    explain_multiclass_record,
)

st.set_page_config(
    page_title="ARGUS | Explainability",
    page_icon="🔍",
    layout="wide"
)

st.title("Model Explainability")
st.caption("Understand why ARGUS classified a network traffic record as a particular class.")

# Model selection
col1, col2 = st.columns(2)

with col1:
    task_label = st.selectbox(
        "Classification Task",
        ["Binary", "Multiclass"]
    )

with col2:
    model_label = st.selectbox(
        "Machine Learning Model",
        ["Logistic Regression", "Random Forest"]
    )

task_key = "binary" if task_label == "Binary" else "multiclass"
model_key = "lr" if model_label == "Logistic Regression" else "rf"

st.divider()

uploaded_file = st.file_uploader(
    "Upload network traffic CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        st.success(f"Loaded {len(data):,} traffic records.")

        with st.expander("Preview uploaded data"):
            st.dataframe(data.head(10), use_container_width=True)

        row_index = st.number_input(
            "Select traffic record (row number)",
            min_value=0,
            max_value=len(data) - 1,
            value=0,
            step=1
        )

        row_index = int(row_index)

        if st.button("Run Prediction and Explain", type="primary"):
            with st.spinner("Running prediction and generating SHAP explanation..."):
                try:
                    predictions = predict_traffic(
                        data,
                        task=task_key,
                        model_type=model_key
                    )

                    prediction = predictions.iloc[row_index]

                    st.subheader("Prediction Result")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Predicted Class",
                            str(prediction["Prediction"])
                        )

                    with c2:
                        st.metric(
                            "Model Confidence",
                            f"{float(prediction['Model Score']):.2%}"
                        )

                    with c3:
                        st.metric(
                            "Model",
                            model_label
                        )

                    if task_key == "binary":
                        explanation = explain_record(
                            data,
                            row_index=row_index,
                            model_type=model_key
                        )
                        explanation_target = "Attack class"
                    else:
                        explanation = explain_multiclass_record(
                            data,
                            row_index=row_index,
                            model_type=model_key
                        )
                        explanation_target = explanation.attrs.get(
                            "predicted_class",
                            str(prediction["Prediction"])
                        )

                    st.divider()
                    st.subheader("SHAP Explanation")

                    st.write(
                        f"Explanation for row **{row_index}** "
                        f"— target: **{explanation_target}**"
                    )

                    st.caption(
                        "Positive SHAP values push the model output toward "
                        "the explained class; negative values push it away. "
                        "These are feature contributions, not causal proof."
                    )

                    if explanation.empty:
                        st.warning("No SHAP contributions were returned.")
                    else:
                        top_features = (
                            explanation
                            .nlargest(15, "Absolute Impact")
                            .sort_values("SHAP Value")
                        )

                        st.subheader("Top 15 Influential Features")
                        st.bar_chart(
                            top_features.set_index("Feature")["SHAP Value"],
                            horizontal=True
                        )

                        st.subheader("Detailed Feature Contributions")
                        st.dataframe(
                            explanation,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download SHAP Explanation CSV",
                            data=explanation.to_csv(index=False).encode("utf-8"),
                            file_name=f"argus_{model_key}_{task_key}_shap_row_{row_index}.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"Prediction or explanation failed: {e}")
                    st.exception(e)

    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.exception(e)

else:
    st.info("Upload a network traffic CSV to begin.")
