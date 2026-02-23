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

import streamlit as st

st.markdown("""
<div class="hero-section">
    <div class="hero-title">Persona-AI</div>
    <div class="hero-subtitle">AI Personality Type Classifier</div>
    <div class="hero-divider"></div>
    <div class="hero-tagline">Transforming Activity Patterns into Intelligent Insights</div>
</div>
""", unsafe_allow_html=True)

# st.markdown("""
# <h1 style='text-align: center;
#            font-size: 48px;
#            background: linear-gradient(90deg,#7F00FF, #E100FF);
#            -webkit-background-clip: text;
#            -webkit-text-fill-color: transparent;
#            font-weight: bold;'>
#  AI Personality Intelligence Dashboard
# </h1>
# """, unsafe_allow_html=True)

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

#######################################################

# --------------------------------------------------
# 🚀 PREMIUM SIDEBAR - Persona-AI
# --------------------------------------------------

st.sidebar.markdown("## 🧠 Persona-AI Dashboard")

st.sidebar.markdown("---")

# 📌 About Section
st.sidebar.markdown("### 📌 About Project")
st.sidebar.write("""
AI-powered personality classification system  
based on behavioral activity metrics.

✔ 9 Input Features  
✔ 4 Personality Classes  
✔ Multiple ML Algorithms  
✔ Real-time Prediction & Analysis
""")

st.sidebar.markdown("---")

# 🤖 Model Information
st.sidebar.markdown("### 🤖 Models Implemented")

model_list = [
    "KNN",
    "Random Forest",
    "SVM",
    "Logistic Regression",
    "XGBoost"
]

for model in model_list:
    st.sidebar.write(f"• {model}")

st.sidebar.markdown("---")

# 📊 Model Ranking (Dynamic)
st.sidebar.markdown("### 📊 Model Performance Ranking")

sorted_models = sorted(accuracy_dict.items(), key=lambda x: x[1], reverse=True)

for i, (name, acc) in enumerate(sorted_models, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔹"
    st.sidebar.write(f"{medal} {name} — {acc*100:.2f}%")

best_model = sorted_models[0][0]

st.sidebar.success(f"🏆 Best Performing Model: {best_model}")

st.sidebar.markdown("---")

# 📂 Dataset Info
st.sidebar.markdown("### 📂 Dataset Information")
st.sidebar.write("""
• Features: Social, Study, Sleep, Group,  
  Productivity, Creativity, Leadership,  
  Decision Speed, Consistency  

• Target Classes:
  - Independent  
  - Analytical  
  - Creative  
  - Leader  

• Preprocessing:
  - StandardScaler  
  - Label Encoding  
""")

st.sidebar.markdown("---")

# ⚙ How It Works (Expandable)
with st.sidebar.expander("⚙ How Persona-AI Works"):
    st.write("""
1️⃣ User enters behavioral data  
2️⃣ Data is standardized using scaler  
3️⃣ Selected ML model performs classification  
4️⃣ Probability score calculated  
5️⃣ AI report generated  
6️⃣ Feature importance analyzed (Random Forest)  
""")

st.sidebar.markdown("---")

# 🎛 Interactive Controls
st.sidebar.markdown("### 🎛 Dashboard Controls")

show_importance = st.sidebar.checkbox("🌲 Show Feature Importance", value=True)
show_accuracy_chart = st.sidebar.checkbox("📈 Show Accuracy Chart", value=True)

st.sidebar.markdown("---")

# 👩‍💻 Developer Section
st.sidebar.markdown("### 👩‍💻 Developer")

st.sidebar.write("""
**Riddhi Dulani**  
B.Tech – Artificial Intelligence & Data Science  
Machine Learning & Behavioral Analytics Enthusiast
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Persona-AI | Advanced ML Deployment")

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
    productivity = st.slider("Productivity Score (1-10)", 1, 10, 5)
    creativity = st.slider("Creativity Score (1-10)", 1, 10, 5)
    leadership = st.slider("Leadership Score (1-10)", 1, 10, 5)
    decision = st.slider("Decision Speed (1-10)", 1, 10, 5)
    consistency = st.slider("Consistency Score (1-10)", 1, 10, 5)

  

    user_data = np.array([[social, study, sleep, group,
                       productivity, creativity,
                       leadership, decision, consistency]])

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

    radar_values = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]


with col2:
    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5,5))
    ax.plot(angles, radar_values, color="#FF4B91", linewidth=3)
    ax.fill(angles, radar_values, color="#FF4B91", alpha=0.3)

    ax.plot(angles, radar_values, linewidth=3)
    ax.fill(angles, radar_values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits)

    ax.set_yticklabels([])
    ax.set_title("Trait Radar Chart", pad=20)
    st.pyplot(fig)

   


# --------------------------------------------------
# 🚀 Prediction Button
# --------------------------------------------------
if st.button("Predict Personality"):
    if "full_report" in st.session_state:
        del st.session_state.full_report

    user_scaled = scaler.transform(user_data)

    # Dynamically pick model selected by user
    model = models[model_choice]

    if model_choice == "XGBoost":
        pred_encoded = model.predict(user_scaled)[0]
        prediction = le.inverse_transform([pred_encoded])[0]
    else:
        prediction = model.predict(user_scaled)[0]

    confidence = model.predict_proba(user_scaled).max() * 100

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

def generate_detailed_report(prediction, confidence, traits, values):

    report = f"""
==============================
 AI PERSONALITY ANALYSIS REPORT
==============================

Predicted Personality Type: {prediction}
Model Confidence Level: {confidence:.2f}%

----------------------------------------
🔎 Confidence Analysis
----------------------------------------
"""

    if confidence >= 80:
        report += "High confidence prediction. Personality traits strongly match model pattern.\n"
    elif confidence >= 60:
        report += "Moderate confidence. Some traits strongly match while others are balanced.\n"
    else:
        report += "Low confidence. Personality traits are mixed or inconsistent.\n"

    report += "\n----------------------------------------\n"
    report += "📊 Trait Analysis & Improvement Guide\n"
    report += "----------------------------------------\n"

    for trait, value in zip(traits, values[:-1]):

        report += f"\n{trait} Score: {value}\n"

        if value <= 4:
            report += "Improvement Needed:\n"
            report += "- Build small daily habits\n"
            report += "- Set weekly improvement goals\n"
            report += "- Track progress consistently\n"

        elif value <= 7:
            report += "Good but can improve further:\n"
            report += "- Practice consistency\n"
            report += "- Take small challenges\n"
            report += "- Reflect weekly\n"

        else:
            report += "Strong trait:\n"
            report += "- Maintain this strength\n"
            report += "- Use it to lead or inspire others\n"

    report += "\n----------------------------------------\n"
    report += "🌟 Positive Strength Summary\n"
    report += "----------------------------------------\n"

    strong_traits = [t for t, v in zip(traits, values[:-1]) if v >= 7]

    if strong_traits:
        report += "Your strongest areas are: " + ", ".join(strong_traits)
    else:
        report += "Balanced personality with growth potential in multiple areas."

    report += "\n\n----------------------------------------\n"
    report += "🚀 Growth Strategy Recommendations\n"
    report += "----------------------------------------\n"
    report += "- Maintain consistent sleep cycle\n"
    report += "- Reduce unnecessary social media time\n"
    report += "- Focus on deep work sessions\n"
    report += "- Practice leadership in small group tasks\n"
    report += "- Improve decision-making with timed tasks\n"

    return report

# --------------------------------------------------
# 📊 RESULT SECTION (Two Column + Radar Chart)
# --------------------------------------------------
if st.session_state.get("prediction_done"):

    

    col1, col2 = st.columns([1, 1])

    summary, rec, hob = sumgen(st.session_state.prediction)

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

    # RIGHT COLUMN (Ai Report)
    with col2:
        st.markdown("### 📄 Detailed AI Report")

        if st.button("📄 Generate Full Personality Report"):

            st.session_state.full_report = generate_detailed_report(
            st.session_state.prediction,
            st.session_state.confidence,
            traits,
            values
        )

    # Show report only if it exists
        if "full_report" in st.session_state:
            st.subheader("📄 Detailed Personality Report")

            st.text_area(
         "Generated Report",
            st.session_state.full_report,
            height=500
            )

            st.download_button(
            label="⬇ Download Report as TXT",
            data=st.session_state.full_report,
            file_name="AI_Personality_Report.txt",
            mime="text/plain"
            )


# --------------------------------------------------
# 🌲 Random Forest Feature Importance
# --------------------------------------------------

st.subheader("🌲 Random Forest Feature Importance")

rf_model = models["Random Forest"]

if hasattr(rf_model, "feature_importances_"):

    feature_names = [
        "Social", "Study", "Sleep", "Group",
        "Productivity", "Creativity",
        "Leadership", "Decision", "Consistency"
    ]

    importances = rf_model.feature_importances_

    # Sort features by importance
    sorted_idx = np.argsort(importances)
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.barh(sorted_features, sorted_importances)
    ax2.set_title("Random Forest Feature Importance")
    ax2.set_xlabel("Importance Score")

    st.pyplot(fig2)

else:
    st.warning("Feature importance not available for this model.")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
---
### 🚀 Persona-AI
Developed using Machine Learning & Behavioral Analytics  
© 2026 Riddhi Dulani | B.Tech AI & DS
""")
