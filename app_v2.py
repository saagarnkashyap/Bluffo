import streamlit as st
import pandas as pd
import joblib
import json
from PIL import Image
import os

# Load model and vectorizer
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("logreg_vectorizer.pkl")

# Page Config
st.set_page_config(page_title="Bluffo: Is This Cap?", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    /* Set dark background for the whole page */
    .main {
        background-color: #000000;
        color: white;
    }

    /* Style sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #111111 !important;
    }

    /* Style buttons */
    .stButton>button {
        background-color: #333333;
        color: white;
        border: 1px solid #555;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #444444;
        color: #00FF99;
    }

    /* Text input boxes and file uploader */
    .stTextInput>div>div>input,
    .stFileUploader>div,
    .stTextArea>div>textarea {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #555555;
    }

    /* Upload file box specific styling */
    .stFileUploader {
        background-color: #1e1e1e;
        border: 1px solid #444;
        padding: 10px;
        border-radius: 5px;
    }

    /* Headings white and bold */
    h1, h2, h3, h4 {
        color: white;
        font-weight: bold;
    }

    /* Meme style main heading */
    .bluffo-title {
        font-size: 60px;
        font-weight: 900;
        text-align: center;
        color: white;
        text-shadow: 3px 3px #ff0000;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #ff4b4b; text-shadow: 3px 3px #000000; font-family: Comic Sans MS, cursive;'>
        Bluffo - Is This Cap?
    </h1>
    """,
    unsafe_allow_html=True
)

# ===== CSV Upload Section at Top =====
st.markdown('<div class="subtitle">📂 Bulk Headline Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column:", type=["csv"])

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    if 'text' in df_uploaded.columns:
        vect_texts = vectorizer.transform(df_uploaded['text'])
        preds = model.predict(vect_texts)
        df_uploaded['Prediction'] = preds
        df_uploaded['Prediction'] = df_uploaded['Prediction'].map({1: 'REAL', 0: 'FAKE'})
        st.dataframe(df_uploaded)

        # Pie Chart
        pie_data = df_uploaded['Prediction'].value_counts()
        st.markdown("###Prediction Distribution")
        st.bar_chart(pie_data)
    else:
        st.error("CSV must contain a 'text' column.")

st.markdown("---")

# ===== Single Prediction Section =====
st.markdown('<div class="subtitle">Analyze a Headline</div>', unsafe_allow_html=True)
user_input = st.text_area("Enter a news headline:", "")

if st.button("🧢 Is This Cap?"):
    if user_input.strip() == "":
        st.warning("Bruh... Type something!")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]
        if prediction == 1:
            st.success("This headline is predicted to be **REAL**.")
        else:
            st.error("This headline is predicted to be **FAKE**.")

# ===== Model Performance =====
st.markdown("---")
st.markdown('<div class="subtitle">Model Performance</div>', unsafe_allow_html=True)

# Layout for side-by-side plots
col1, col2, col3 = st.columns(3)

if os.path.exists("confusion_matrix_v3.png"):
    with col1:
        st.markdown("**Confusion Matrix**")
        st.image("confusion_matrix_v3.png", use_container_width=True)

if os.path.exists("roc_curve_v3.png"):
    with col2:
        st.markdown("**ROC Curve**")
        st.image("roc_curve_v3.png", use_container_width=True)

if os.path.exists("precision_recall_v3.png"):
    with col3:
        st.markdown("**Precision-Recall Curve**")
        st.image("precision_recall_v3.png", use_container_width=True)

# ===== Classification Report =====
if os.path.exists("classification_metrics_v3.json"):
    st.markdown("**Classification Metrics**")
    with open("classification_metrics_v3.json", "r") as f:
        report_data = json.load(f)
        df_report = pd.DataFrame(report_data).transpose().round(2)
        df_report = df_report.dropna(axis=1, how="all")
        st.dataframe(df_report.style.background_gradient(cmap="Greens"), use_container_width=True)
