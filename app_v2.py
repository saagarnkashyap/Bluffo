import streamlit as st
import joblib
import os
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load model and vectorizer
vectorizer = joblib.load("vectorizer_v3.joblib")
vectorizer = joblib.load("logreg_vectorizer.pkl")

# Page Setup
st.set_page_config(page_title="Bluffo - Fake News Detection", layout="wide")

# Style Overrides
st.markdown("""
    <style>
        .main {
            background-color: #1c48ec;
        }
        .title {
            font-size: 48px;
            font-weight: 900;
            color: white;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 22px;
            margin-top: 20px;
            color: #f5f5f5;
        }
        .stTextArea label {
            font-weight: bold;
            color: white;
        }
        .stMarkdown, .stDataFrame div {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">Bluffo - Fake News Detection</div>', unsafe_allow_html=True)
st.write("Enter a news headline below to check if it's Real or Fake.")

# Input
user_input = st.text_area("News Headline", "")

# Prediction
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a news headline.")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]

        if prediction == 1:
            st.success("This news is predicted to be REAL.")
        else:
            st.error("This news is predicted to be FAKE.")

# Separator
st.markdown("---")
st.markdown('<div class="subtitle">Model Performance</div>', unsafe_allow_html=True)

# Performance Images
if os.path.exists("roc_curve_v3.png"):
    st.markdown("**ROC Curve**")
    st.image(Image.open("roc_curve_v3.png"), use_container_width=True)

if os.path.exists("confusion_matrix_v3.png"):
    st.markdown("**Confusion Matrix**")
    st.image(Image.open("confusion_matrix_v3.png"), use_container_width=True)

if os.path.exists("precision_recall_v3.png"):
    st.markdown("**Precision-Recall Curve**")
    st.image(Image.open("precision_recall_v3.png"), use_container_width=True)

# Classification Report Table
if os.path.exists("classification_metrics_v3.json"):
    st.markdown("**Classification Report**")
    with open("classification_metrics_v3.json", "r") as f:
        report_data = json.load(f)
        df_report = pd.DataFrame(report_data).transpose().round(2)
        df_report = df_report.dropna(axis=1, how="all")
        st.dataframe(df_report.style.background_gradient(cmap="Blues"), use_container_width=True)

# CSV Upload Option
st.markdown("---")
st.markdown('<div class="subtitle">Upload CSV File</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a CSV file with a column named 'Headline'", type=["csv"])

if uploaded_file:
    df_uploaded = pd.read_csv(uploaded_file)
    if "Headline" in df_uploaded.columns:
        df_uploaded["Prediction"] = model.predict(vectorizer.transform(df_uploaded["Headline"].astype(str)))
        st.success("Predictions generated!")

        # Pie chart
        counts = df_uploaded["Prediction"].value_counts().sort_index()
        labels = ["Fake", "Real"]
        plt.figure(figsize=(4, 4))
        plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"])
        plt.title("Prediction Distribution")
        st.pyplot(plt.gcf())

        st.dataframe(df_uploaded, use_container_width=True)
    else:
        st.error("CSV must contain a column named 'Headline'.")
