import streamlit as st
import pickle
import os
from PIL import Image
import json
import pandas as pd

# Load model and vectorizer
with open("fake_news_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer_v2.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page Setup
st.set_page_config(page_title="Bluffo - Fake News Detection", layout="wide")

# Style Overrides
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #1c48ec;
        }
        h1 {
            font-size: 48px;
            color: white;
            font-weight: bold;
        }
        .subtitle {
            font-size: 22px;
            margin-top: 20px;
            color: #f5f5f5;
            text-align: center;
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
st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <h1 style="font-size: 48px; color: white; font-weight: 900;">Bluffo - Fake News Detection</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="subtitle">Enter a news headline below to check if it\'s Real or Fake.</div>', unsafe_allow_html=True)


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

# --- CSV Upload Section ---
st.markdown("---")
st.markdown('<div class="subtitle">Bulk Prediction via CSV</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file with a 'Headline' column", type=["csv"])

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)

        if 'Headline' not in df_uploaded.columns:
            st.error("CSV must contain a 'Headline' column.")
        else:
            st.write("Preview of Uploaded File:")
            st.dataframe(df_uploaded.head(), use_container_width=True)

            if st.button("Predict All Headlines"):
                vect_inputs = vectorizer.transform(df_uploaded['Headline'].astype(str))
                predictions = model.predict(vect_inputs)
                df_uploaded["Prediction"] = ["REAL" if p == 1 else "FAKE" for p in predictions]

                st.markdown("**Prediction Results:**")
                st.dataframe(df_uploaded[["Headline", "Prediction"]], use_container_width=True)

                # Pie Chart
                st.markdown("**Prediction Summary**")
                summary = df_uploaded["Prediction"].value_counts().reset_index()
                summary.columns = ["Label", "Count"]

                fig = px.pie(
                    summary, names="Label", values="Count",
                    color="Label",
                    color_discrete_map={"REAL": "lightblue", "FAKE": "crimson"},
                    hole=0.3
                )
                fig.update_traces(textinfo="percent+label", textfont_size=16)
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Something went wrong: {e}")


# Separator
st.markdown("---")
st.markdown('<div class="subtitle">Model Performance</div>', unsafe_allow_html=True)

# ROC Curve
if os.path.exists("roc_curve_v2.png"):
    st.markdown("**ROC Curve**")
    st.image(Image.open("roc_curve_v2.png"), use_container_width=True)

# Confusion Matrix
if os.path.exists("confusion_matrix_v2.png"):
    st.markdown("**Confusion Matrix**")
    st.image(Image.open("confusion_matrix_v2.png"), use_container_width=True)

# Precision-Recall Curve (new)
if os.path.exists("precision_recall_v2.png"):
    st.markdown("**Precision-Recall Curve**")
    st.image(Image.open("precision_recall_v2.png"), use_container_width=True)

# Classification Report Table (styled)
if os.path.exists("classification_metrics_v2.json"):
    st.markdown("**Classification Report**")
    with open("classification_metrics_v2.json", "r") as f:
        report_data = json.load(f)
        df_report = pd.DataFrame(report_data).transpose().round(2)
        df_report = df_report.dropna(axis=1, how="all")  # clean unnecessary columns
        st.dataframe(df_report.style.background_gradient(cmap="Blues"), use_container_width=True)
