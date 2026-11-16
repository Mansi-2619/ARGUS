import streamlit as st
import pandas as pd
from pathlib import Path

from src.explainability import explain_record


st.title("AI Explainability")
st.caption("ARGUS | Understanding why the model made a prediction")

sample_path = Path("D:/ARGUS/data/sample_traffic.csv")

uploaded = st.file_uploader(
    "Upload traffic CSV for explanation",
    type=["csv"]
)

if uploaded is not None:
    data = pd.read_csv(uploaded, low_memory=False)
elif sample_path.exists():
    data = pd.read_csv(sample_path, low_memory=False)
    st.info("Using the existing 100-record sample traffic dataset.")
else:
    st.warning("Upload a compatible traffic CSV to continue.")
    st.stop()

st.metric("Available traffic records", f"{len(data):,}")

row_index = st.slider(
    "Select traffic record",
    min_value=0,
    max_value=len(data) - 1,
    value=0
)

if st.button("Explain selected prediction", type="primary"):
    try:
        with st.spinner("Calculating SHAP feature contributions..."):
            explanation = explain_record(data, row_index=row_index)

        st.session_state["argus_shap"] = explanation
        st.session_state["argus_shap_row"] = row_index

    except Exception as error:
        st.error(f"Explanation failed: {error}")

if (
    "argus_shap" in st.session_state
    and st.session_state.get("argus_shap_row") == row_index
):
    explanation = st.session_state["argus_shap"]

    st.divider()
    st.subheader("Feature contribution analysis")

    top_features = explanation.head(10).copy()

    st.write("### Top 10 influential features")
    chart_data = top_features.set_index("Feature")["SHAP Value"]
    st.bar_chart(chart_data, horizontal=True)

    st.write("### Detailed explanation")

    st.dataframe(
        top_features[
            ["Feature", "SHAP Value", "Direction", "Absolute Impact"]
        ],
        use_container_width=True,
        hide_index=True
    )

    attack_features = int(
        (explanation["SHAP Value"] > 0).sum()
    )
    benign_features = int(
        (explanation["SHAP Value"] < 0).sum()
    )

    col1, col2 = st.columns(2)
    col1.metric("Features pushing towards Attack", attack_features)
    col2.metric("Features pushing towards Benign", benign_features)

    st.caption(
        "SHAP values describe the influence of features on the model output "
        "relative to the selected background dataset. They do not establish "
        "that a traffic record is malicious."
    )

    st.download_button(
        "Download SHAP explanation",
        data=explanation.to_csv(index=False).encode("utf-8"),
        file_name=f"argus_shap_record_{row_index + 1}.csv",
        mime="text/csv"
    )
