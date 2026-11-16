\# ARGUS

\### Explainable AI-Based Cyber Threat Detection Platform



ARGUS is a machine-learning-based Intrusion Detection System (IDS) that analyzes network traffic, classifies potential cyber threats, and explains individual predictions using SHAP.



\## Features



\- Binary classification: Benign vs Attack

\- Multiclass classification: 15 traffic classes

\- Logistic Regression and Random Forest models

\- CSV-based network traffic analysis

\- Prediction confidence and traffic distribution

\- SHAP-based individual prediction explanations

\- Binary and multiclass model comparison

\- Confusion matrices and classification reports

\- Downloadable prediction and evaluation results



\## Technology Stack



\- Python

\- Streamlit

\- Scikit-learn

\- Pandas and NumPy

\- SHAP

\- Matplotlib

\- Joblib



\## Project Structure



```text

ARGUS/

├── artifacts/

│   ├── models/

│   ├── preprocessing/

│   └── evaluation/

├── data/

│   ├── raw/

│   └── sample\_traffic.csv

├── pages/

│   ├── 1\_Traffic\_Predictions.py

│   ├── 2\_Explainability.py

│   └── 3\_Model\_Comparison.py

├── src/

│   ├── config.py

│   ├── load\_data.py

│   ├── preprocess.py

│   ├── inference.py

│   └── explainability.py

├── app.py

├── requirements.txt

└── scripts\_evaluate.py

