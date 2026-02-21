import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# 🎨 Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Personality Classifier",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# 🎨 Load External CSS
# --------------------------------------------------
def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --------------------------------------------------
# 🧠 Load Saved Models
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
    accuracy_dict = joblib.load("saved_models/model_accuracies.pkl")

    return models, scaler, le, accuracy_dict


models, scaler, le, accuracy_dict = load_models()

# --------------------------------------------------
# 📊 Model Accuracy Overview
# --------------------------------------------------
st.subheader("📊 Model Performance Overview")

cols = st.columns(len(accuracy_dict))

for col, (name, acc) in zip(cols, accuracy_dict.items()):
    col.metric(name, f"{acc*100:.2f}%")

best_model = max(accuracy_dict, key=accuracy_dict.get)
st.success(f"🏆 Best Performing Model: {best_model}")

# --------------------------------------------------
# 🔍 Model Selection
# --------------------------------------------------
st.subheader("🔍 Select Model for Prediction")

model_choice = st.selectbox(
    "Choose Model",
    list(models.keys())
)

# --------------------------------------------------
# 📥 User Input Section
# --------------------------------------------------
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

# --------------------------------------------------
# 🚀 Prediction Button
# --------------------------------------------------
if st.button("🚀 Predict Personality"):

    user_scaled = scaler.transform(user_data)
    model = models[model_choice]

    if model_choice == "XGBoost":
        pred_encoded = model.predict(user_scaled)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
        confidence = np.max(model.predict_proba(user_scaled)) * 100
    else:
        prediction = model.predict(user_scaled)[0]
        confidence = np.max(model.predict_proba(user_scaled)) * 100

    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.confidence = confidence

# --------------------------------------------------
# 🎯 Personality Summary Function
# --------------------------------------------------
def sumgen(prediction):

    if prediction == "Independent":
        summary = "🌿 You are self-driven and value independence."
        rec = "Best suited for roles with autonomy."
        hob = "Solo travel ✈️ | Freelancing 💻 | Blogging ✍️ | Fitness 🏋️"

    elif prediction == "Analytical":
        summary = "📊 Logical and detail-oriented personality."
        rec = "Excellent for research and strategy roles."
        hob = "Chess ♟️ | Coding 💻 | Stock analysis 📈 | Puzzles 🧩"

    elif prediction == "Creative":
        summary = "🎨 Imaginative and innovative thinker."
        rec = "Great for design and creative industries."
        hob = "Photography 📸 | Music 🎶 | Content creation 🎥"

    elif prediction == "Leader":
        summary = "👑 Confident and team-oriented personality."
        rec = "Strong fit for managerial positions."
        hob = "Entrepreneurship 🚀 | Debate 🎙️ | Event management 🎤"

    return summary, rec, hob

# --------------------------------------------------
# 📊 RESULT SECTION (Two Column + Radar Chart)
# --------------------------------------------------
if st.session_state.get("prediction_done"):

    summary, rec, hob = sumgen(st.session_state.prediction)

    traits = [
        "Social", "Study", "Sleep", "Group",
        "Productivity", "Creativity",
        "Leadership", "Decision", "Consistency"
    ]

    values = [
        social, study, sleep, group,
        productivity, creativity,
        leadership, decision, consistency
    ]

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]

    col1, col2 = st.columns([1, 1])

    # LEFT COLUMN (Prediction Card)
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #E06AA1, #6E1A7A);
            padding: 25px;
            border-radius: 18px;
            color: white;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        ">
            <h2>🎯 Predicted Personality</h2>
            <h3>{st.session_state.prediction}</h3>
            <p style="font-size:18px;">
                📊 Confidence Score: <b>{st.session_state.confidence:.2f}%</b>
            </p>
            <hr>
            <p>{summary}</p>
            <p><b>Recommendation:</b> {rec}</p>
            <p><b>Suggested Hobbies:</b><br>{hob}</p>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT COLUMN (Radar Chart)
    with col2:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5,5))

        ax.plot(angles, values, linewidth=3)
        ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(traits)

        ax.set_yticklabels([])
        ax.set_title("Trait Radar Chart", pad=20)

        st.pyplot(fig)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Advanced AI/ML Personality Classification Project | Production Ready Deployment")
