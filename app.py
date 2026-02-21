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
    accuracy_dict = joblib.load("saved_models/model_accuracies.pkl")

    return models, scaler, le, accuracy_dict


models, scaler, le, accuracy_dict = load_models()

st.subheader("📊 Model Accuracy Overview")

for name, acc in accuracy_dict.items():
    st.metric(name, f"{acc*100:.2f}%")

best_model = max(accuracy_dict, key=accuracy_dict.get)
st.success(f"🏆 Best Model: {best_model}")

#####################################################################################

st.subheader("🔍 Select Model")

model_choice = st.selectbox(
    "Choose Model for Prediction",
    list(models.keys())
)

st.subheader("📥 Enter Your Activity Details")

col1, col2 = st.columns(2)

with col1:
    social = st.slider("Social Media Hours", 0.0, 10.0, 2.0)
    study = st.slider("Study Hours", 0.0, 10.0, 4.0)
    sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
    group = st.slider("Group Preference (1-5)", 1, 5, 3)

with col2:
    productivity = st.slider("Productivity Score (1-10)", 1, 10, 5)
    creativity = st.slider("Creativity Score (1-10)", 1, 10, 5)
    leadership = st.slider("Leadership Score (1-10)", 1, 10, 5)
    decision = st.slider("Decision Speed (1-10)", 1, 10, 5)
    consistency = st.slider("Consistency Score (1-10)", 1, 10, 5)

user_data = np.array([[social, study, sleep, group,
                       productivity, creativity,
                       leadership, decision, consistency]])


if st.button("🚀 Predict Personality"):

    # Scale input
    user_scaled = scaler.transform(user_data)

    model = models[model_choice]

    if model_choice == "XGBoost":
        pred_encoded = model.predict(user_scaled)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
        confidence = np.max(model.predict_proba(user_scaled)) * 100
    else:
        prediction = model.predict(user_scaled)[0]
        confidence = np.max(model.predict_proba(user_scaled)) * 100

    # Store in session
    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.confidence = confidence

###############################################
def sumgen(prediction):

    if prediction == "Independent":
        summary = "🌿 You are self-driven and value independence."
        rec = "Best in roles with autonomy."
        hob = "Solo travel ✈️ | Freelancing 💻 | Blogging ✍️ | Fitness 🏋️"

    elif prediction == "Analytical":
        summary = "📊 Logical and detail-oriented personality."
        rec = "Great for research and strategy roles."
        hob = "Chess ♟️ | Coding 💻 | Stock analysis 📈 | Puzzles 🧩"

    elif prediction == "Creative":
        summary = "🎨 Imaginative and innovative thinker."
        rec = "Excel in design and creative fields."
        hob = "Photography 📸 | Content creation 🎥 | Music 🎶 | Fashion 👗"

    elif prediction == "Leader":
        summary = "👑 Confident and team-oriented personality."
        rec = "Strong in managerial roles."
        hob = "Entrepreneurship 🚀 | Debate 🎙️ | Event management 🎤"

    return summary, rec, hob

if st.session_state.get("prediction_done"):

    summary, rec, hob = sumgen(st.session_state.prediction)

    st.markdown("### 🎯 Predicted Personality")

    st.success(f"""
    **Personality:** {st.session_state.prediction}  
    **Confidence:** {st.session_state.confidence:.2f}%
    """)

    st.write(summary)
    st.write("**Recommendation:**", rec)
    st.write("**Suggested Hobbies:**", hob)
