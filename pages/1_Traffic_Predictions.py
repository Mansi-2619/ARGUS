from src.explainability import (
    explain_record,
    explain_multiclass_record
)
import streamlit as st
import pandas as pd

from src.inference import predict_traffic


st.title("Traffic Predictions")
st.caption("ARGUS | Explainable AI-Based Cyber Threat Detection Platform")

st.info(
    "Upload a CSV containing the original 78 CICIDS2017 traffic features. "
    "ARGUS will apply the saved research preprocessing and Logistic Regression model."
)

task = st.selectbox(
    "Choose detection mode",
    ["Binary Detection", "Multiclass Detection"]
)

task_key = "binary" if task == "Binary Detection" else "multiclass"

uploaded_file = st.file_uploader(
    "Upload network traffic CSV",
    type=["csv"],
    help="The CSV must contain all 78 features expected by the selected model."
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, low_memory=False)

        st.write("### Uploaded dataset")
        col1, col2 = st.columns(2)
        col1.metric("Traffic records", f"{len(data):,}")
        col2.metric("CSV columns", f"{len(data.columns):,}")

        with st.expander("Preview uploaded data"):
            st.dataframe(data.head(10), use_container_width=True)

        if st.button("Run Threat Detection", type="primary", use_container_width=True):
            with st.spinner("Processing traffic and generating predictions..."):
                predictions = predict_traffic(data, task_key)

                st.session_state["argus_predictions"] = predictions
                st.session_state["argus_input_data"] = data.reset_index(drop=True).copy()
                st.session_state["argus_export"] = pd.concat(
                    [
                        data.reset_index(drop=True),
                        predictions.drop(
                            columns=["Row", "Model"], errors="ignore"
                        ).reset_index(drop=True)
                    ],
                    axis=1
                )
                st.session_state["argus_task"] = task_key

            st.success("Prediction completed successfully.")

    except Exception as error:
        st.error(f"Unable to process this CSV: {error}")

if (
    "argus_predictions" in st.session_state
    and st.session_state.get("argus_task") == task_key
):
    results = st.session_state["argus_predictions"]

    st.divider()
    st.header("Detection Results")

    total = len(results)
    
    if task_key == "binary":
            attack_count = int((results["Prediction"] == "Attack").sum())
    else:
            benign_mask = (
                results["Prediction"]
                .astype(str)
                .str.strip()
                .str.upper()
                .eq("BENIGN")
            )
            attack_count = int((~benign_mask).sum())


    benign_count = total - attack_count
    avg_score = results["Model Score"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{total:,}")
    c2.metric("Predicted attacks", f"{attack_count:,}")
    c3.metric("Predicted benign", f"{benign_count:,}")
    c4.metric("Average model score", f"{avg_score:.2f}%")

    st.caption(
        "Model score represents prediction confidence, not verified accuracy. "
        "A predicted attack is not by itself confirmation of a real incident."
    )

    tab1, tab2 = st.tabs(["Prediction Overview", "Detailed Results"])

    with tab1:
        st.subheader("Predicted traffic distribution")
        distribution = (
            results["Prediction"]
            .value_counts()
            .rename_axis("Predicted class")
            .reset_index(name="Records")
            .set_index("Predicted class")
        )
        st.bar_chart(distribution)

    with tab2:
        st.subheader("Traffic-level predictions")
        display_results = results.copy()
        display_results["Model Score"] = (
            display_results["Model Score"] * 100
        ).round(2).astype(str) + "%"

        if "Attack Score" in display_results.columns:
            display_results["Attack Score"] = (
                display_results["Attack Score"] * 100
            ).round(2).astype(str) + "%"

        st.dataframe(display_results, use_container_width=True, height=450)

    st.divider()
    st.header("Explain a Prediction")

    original_data = st.session_state["argus_input_data"]

    selected_row = st.number_input(
        "Select prediction row",
        min_value=0,
        max_value=len(results) - 1,
        value=0,
        step=1,
        key=f"argus_shap_row_{task_key}"
    )

    selected_row = int(selected_row)
    selected_prediction = results.iloc[selected_row]["Prediction"]

    st.write(f"**Model prediction:** {selected_prediction}")

    if st.button("Explain Selected Prediction", type="primary"):
        with st.spinner("Calculating SHAP feature contributions..."):
            if task_key == "binary":
                explanation = explain_record(
                    original_data,
                    row_index=selected_row
                )
                explanation_title = "Binary explanation"
            else:
                explanation = explain_multiclass_record(
                    original_data,
                    row_index=selected_row
                )
                explanation_title = (
                    f"Class-specific explanation: "
                    f"{explanation.attrs['predicted_class']}"
                )

        st.subheader(explanation_title)

        st.caption(
            "Positive SHAP values push towards the explained class; "
            "negative values push away from it."
        )

        st.subheader("Top contributing features")

        chart_data = (
            explanation.head(10)
            .set_index("Feature")[["SHAP Value"]]
        )
        st.bar_chart(chart_data)

        st.subheader("Feature contribution details")

        st.dataframe(
            explanation.head(10),
            use_container_width=True
        )

        explanation_csv = explanation.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download SHAP Explanation",
            data=explanation_csv,
            file_name=f"argus_{task_key}_shap_row_{selected_row}.csv",
            mime="text/csv"
        )

        st.caption(
            "SHAP explains the model's decision using feature contributions. "
            "It does not prove malicious activity or confirm a security incident."
        )
  
    export_data = st.session_state["argus_export"]
    csv_bytes = export_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="argus_traffic_predictions.csv",
        mime="text/csv",
        type="primary"
    )
else:
    st.divider()
    st.subheader("Ready for traffic analysis")
    st.write(
        "Upload a compatible CSV and select **Run Threat Detection** "
        "to generate predictions."
    )
