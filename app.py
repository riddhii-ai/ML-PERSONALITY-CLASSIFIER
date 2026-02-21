import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.set_page_config(
    page_title="AI Personality Classifier",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<h1 style='text-align: center;
           font-size: 48px;
           background: linear-gradient(90deg,#0CDDEC, #FEC700);
           -webkit-background-clip: text;
           -webkit-text-fill-color: transparent;
           font-weight: bold;'>
 AI Personality Intelligence Dashboard
</h1>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Saved Models
# --------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "KNN": joblib.load("saved_models/KNN.pkl"),
        "Random Forest": joblib.load("saved_models/Random_Forest.pkl"),
        "SVM": joblib.load("saved_models/SVM.pkl"),
        "Logistic Regression": joblib.load("saved_models/Logistic_Regression.pkl"),
        "XGBoost": joblib.load("saved_models/XGBoost.pkl"),
    }

    scaler = joblib.load("saved_models/scaler.pkl")
    le = joblib.load("saved_models/label_encoder.pkl")

    return models, scaler, le

models, scaler, le = load_models()
