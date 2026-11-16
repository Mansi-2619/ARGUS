
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ARGUS | Cyber Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

[data-testid="stSidebar"] {
    background: #101d35;
}

[data-testid="stSidebar"] * {
    color: #e8eef9 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #34435c;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.argus-brand {
    font-size: 27px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 2px;
}

.argus-sub {
    color: #9eacc3;
    font-size: 11px;
    letter-spacing: 2px;
}

.page-title {
    font-size: 30px;
    font-weight: 800;
    color: #14233f;
}

.page-subtitle {
    color: #718096;
    font-size: 14px;
}

.metric-card {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.metric-value {
    color: #172b4d;
    font-size: 27px;
    font-weight: 800;
    margin-top: 9px;
}

.section-card {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 12px;
    padding: 22px;
}

.status-pending {
    color: #9a6700;
    background: #fff6db;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}

.small-muted {
    color: #718096;
    font-size: 12px;
}

div[data-testid="stFileUploader"] {
    background: white;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown('<div class="argus-brand">🛡 ARGUS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="argus-sub">CYBER THREAT INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Traffic Analysis",
            "Upload Traffic",
            "Model Registry",
            "About ARGUS"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")
    st.caption("Dashboard: Online")
    st.caption("Inference engine: Not connected")
    st.caption("SHAP: Not connected")

    st.divider()
    st.caption("ARGUS | Academic Research Project")
    st.caption("CICIDS2017-based IDS")


# ---------- REUSABLE COMPONENTS ----------
def heading(title, subtitle):
    st.markdown(
        f'<div class="page-title">{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )
    st.write("")


def metric(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="small-muted">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------- OVERVIEW ----------
if page == "Overview":

    heading(
        "Security Overview",
        "Explainable AI-based network intrusion detection and investigation."
    )

    a, b, c, d = st.columns(4)

    with a:
        metric("Detection engine", "Pending", "Model integration")
    with b:
        metric("ML architectures", "2", "Logistic Regression · Random Forest")
    with c:
        metric("Classification modes", "2", "Binary · Multiclass")
    with d:
        metric("Input features", "78", "Expected model feature schema")

    st.write("")

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Platform Overview")
        st.write(
            "ARGUS analyses network traffic using saved machine learning "
            "models to identify potentially malicious activity."
        )
        st.write(
            "The platform is being developed around the existing "
            "CICIDS2017 research pipeline, with explainability and "
            "investigation capabilities."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Development Status")
        st.markdown("🟢 Dashboard interface")
        st.markdown("🟡 Model integration")
        st.markdown("🟡 CSV inference")
        st.markdown("🟡 SHAP explanations")
        st.markdown("🟡 Investigation reports")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Core Capabilities")

    x, y, z = st.columns(3)

    with x:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📡 Traffic Analysis")
        st.write("Inspect uploaded network traffic and classification results.")
        st.markdown("</div>", unsafe_allow_html=True)

    with y:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI Detection")
        st.write("Use saved Logistic Regression and Random Forest models.")
        st.markdown("</div>", unsafe_allow_html=True)

    with z:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Explainable AI")
        st.write("Understand model decisions using SHAP explanations.")
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- UPLOAD ----------
elif page == "Upload Traffic":

    heading(
        "Upload Network Traffic",
        "Import a CSV dataset for schema inspection and subsequent analysis."
    )

    uploaded_file = st.file_uploader(
        "Choose a network traffic CSV file",
        type=["csv"],
        help="The inference pipeline will require the expected 78 model features."
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file, nrows=10000)

            st.success(f"File loaded: {uploaded_file.name}")

            a, b, c = st.columns(3)

            with a:
                metric("Rows inspected", f"{len(df):,}", "First 10,000 rows maximum")
            with b:
                metric("Columns", str(len(df.columns)), "Detected CSV columns")
            with c:
                metric("Schema check", "Pending", "Model compatibility not yet checked")

            st.write("")
            st.subheader("Traffic Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Column Information")

            info = pd.DataFrame({
                "Feature": df.columns,
                "Data Type": [str(x) for x in df.dtypes],
                "Missing Values": [int(x) for x in df.isna().sum().values]
            })

            st.dataframe(info, use_container_width=True)

            st.info(
                "This is currently a CSV inspection feature. "
                "Predictions will be enabled after connecting the verified "
                "preprocessing pipeline and saved models."
            )

        except Exception as e:
            st.error(f"Unable to read CSV: {e}")


# ---------- TRAFFIC ANALYSIS ----------
elif page == "Traffic Analysis":

    heading(
        "Traffic Analysis",
        "Network traffic classification and threat investigation."
    )

    st.info(
        "No prediction results are available yet. "
        "The real inference engine will populate this page after integration."
    )

    st.markdown("### Planned analysis outputs")

    st.write("- Benign vs Attack classification")
    st.write("- Multiclass attack category")
    st.write("- Prediction confidence")
    st.write("- Traffic-level investigation")
    st.write("- SHAP-based feature contributions")


# ---------- MODEL REGISTRY ----------
elif page == "Model Registry":

    heading(
        "Model Registry",
        "Saved research models intended for ARGUS integration."
    )

    models = pd.DataFrame([
        ["Logistic Regression", "Binary", "Saved"],
        ["Logistic Regression", "Multiclass", "Saved"],
        ["Random Forest", "Binary", "Candidate — provenance to confirm"],
        ["Random Forest", "Multiclass", "Saved"]
    ], columns=["Architecture", "Task", "Artifact Status"])

    st.dataframe(models, use_container_width=True, hide_index=True)

    st.warning(
        "Artifact presence does not by itself establish the final model's "
        "research provenance. Binary Random Forest selection remains to be confirmed."
    )


# ---------- ABOUT ----------
elif page == "About ARGUS":

    heading(
        "About ARGUS",
        "Explainable AI-Based Cyber Threat Detection Platform"
    )

    st.markdown("""
    ### Project objective

    ARGUS is an academic intrusion detection platform built around
    machine learning models developed using the CICIDS2017 dataset.

    ### Research components

    - Logistic Regression
    - Random Forest
    - Binary classification
    - 15-class multiclass classification
    - SHAP explainability

    ### Application roadmap

    1. Network traffic CSV ingestion
    2. Research-compatible preprocessing
    3. Saved model inference
    4. Explainable predictions
    5. Investigation dashboard and reporting

    **Important:** ARGUS predictions are model estimates, not independently
    confirmed security incidents.
    """)
