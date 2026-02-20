import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --------------------------------------------------
# 🎨 Custom Premium UI Styling
# --------------------------------------------------
# config

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

st.markdown("""
<style>

.stApp {
    background: linear-gradient(30deg, #510A32, #2D142C);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

.stButton>button {
    background: linear-gradient(180deg, #C72C41, #801336);
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);      /* enlarge effect */
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);  /* glow/shadow effect */
    cursor: pointer;
}
        
.stButton>button:active {
    transform: scale(0.97);
}


div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 📂 1. Load Dataset  
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("realistic_personality_dataset.csv")

df = load_data()

X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------------------------------
# 2. Train Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X, y, y_encoded, test_size=0.2, random_state=42
)


# --------------------------------------------------
# 3. Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# X_train_scaled_enc = scaler.fit_transform(X_train_enc)
# X_test_scaled_enc = scaler.transform(X_test_enc)

# --------------------------------------------------
# 🔹4. Train Models
# --------------------------------------------------

@st.cache_resource
def train_models(X_train_scaled, y_train, y_train_enc):
    
    # KNN
    knn = KNeighborsClassifier()
    params = {"n_neighbors": list(range(3, 15))}
    grid = GridSearchCV(knn, params, cv=5)
    grid.fit(X_train_scaled, y_train)
    best_knn = grid.best_estimator_

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train_scaled, y_train)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train_scaled, y_train_enc)

    return best_knn, rf, svm, lr, xgb

best_knn, rf, svm, lr, xgb = train_models(
    X_train_scaled,
    y_train,
    y_train_enc
)


# --------------------------------------------------
# 📊 5. Calculate Accuracy
# --------------------------------------------------
def calculate_accuracy(best_knn,
        rf,
        svm,
        lr,
        xgb,
        X_test_scaled,
        y_test,
        y_test_enc):

    knn_acc = accuracy_score(y_test, best_knn.predict(X_test_scaled))
    rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
    xgb_acc = accuracy_score(y_test_enc, xgb.predict(X_test_scaled))

    return knn_acc, rf_acc, svm_acc, lr_acc, xgb_acc

knn_acc, rf_acc, svm_acc, lr_acc, xgb_acc = calculate_accuracy(best_knn,
        rf,
        svm,
        lr,
        xgb,
        X_test_scaled,
        y_test,
        y_test_enc)



# --------------------------------------------------
# 6. Store All Models in Dictionary
# --------------------------------------------------

models = {
    "KNN": best_knn,
    "Random Forest": rf,
    "SVM": svm,
    "Logistic Regression": lr,
    "XGBoost": xgb
}

# --------------------------------------------------
# 7. Save Models
# --------------------------------------------------

# if not os.path.exists("saved_models"):
#     os.makedirs("saved_models")

# # Check if models already saved
# model_files = [
#     "KNN.pkl",
#     "Random_Forest.pkl",
#     "SVM.pkl",
#     "Logistic_Regression.pkl",
#     "XGBoost.pkl"
# ]

# models_exist = all(
#     os.path.exists(f"saved_models/{file}") for file in model_files
# )

# if not models_exist:
#     for name, model in models.items():
#         file_name = f"saved_models/{name.replace(' ', '_')}.pkl"
#         joblib.dump(model, file_name)

#     joblib.dump(scaler, "saved_models/scaler.pkl")
#     st.sidebar.success("Models Saved Successfully!")
# else:
#     st.sidebar.info("Models already saved")
st.sidebar.info("Models already saved")

# --------------------------------------------------
# 📊 Accuracy Comparison
# --------------------------------------------------
st.subheader("📊 Model Performance Overview")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("KNN", f"{round(knn_acc*100,2)}%")
col2.metric("Random Forest", f"{round(rf_acc*100,2)}%")
col3.metric("SVM", f"{round(svm_acc*100,2)}%")
col4.metric("Logistic Regression", f"{round(lr_acc*100,2)}%")
col5.metric("XGBoost", f"{round(xgb_acc*100,2)}%")

accuracy_dict = {
    "KNN": knn_acc,
    "Random Forest": rf_acc,
    "SVM": svm_acc,
    "Logistic Regression": lr_acc,
    "XGBoost": xgb_acc
}

accuracy_df = pd.DataFrame({
    "Model": accuracy_dict.keys(),
    "Accuracy (%)": [round(v*100,2) for v in accuracy_dict.values()]
})

st.subheader("📊 Model Accuracy Comparison")
st.data_editor(
    accuracy_df,
    use_container_width=True,
    disabled=True
)



# -------------------------------------------------
# charts
# ------------------------------------------------



# # Accuracy Bar Chart
# fig, ax = plt.subplots()
# ax.bar(accuracy_dict.keys(), [v*100 for v in accuracy_dict.values()])
# ax.set_ylabel("Accuracy (%)")
# ax.set_title("Model Performance Comparison")
# plt.xticks(rotation=45)
# st.pyplot(fig)

best_model_name = max(accuracy_dict, key=accuracy_dict.get)
st.success(f"🏆 Best Performing Model: {best_model_name}")

# --------------------------------------------------
# 📌 Sidebar
# --------------------------------------------------
st.sidebar.title("📌 Project Information")
st.sidebar.write("""
This AI system predicts personality type 
based on behavioral activity data.

Models Used:
- KNN
- Random Forest
- SVM
- Logistic Regression
- XGBoost
""")
st.sidebar.success("Best Model: " + best_model_name)



# --------------------------------------------------
# 🔍 Model Selection
# --------------------------------------------------
model_choice = st.selectbox("Select Model for Prediction", list(accuracy_dict.keys()))

# --------------------------------------------------
# 📝 User Input (2 Column Layout)
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
# 🚀 Prediction
# --------------------------------------------------
# 🔘 Predict Button
if st.button("Predict Personality"):

    user_scaled = scaler.transform(user_data)

    if model_choice == "KNN":
        prediction = best_knn.predict(user_scaled)[0]
        confidence = np.max(best_knn.predict_proba(user_scaled)) * 100

    elif model_choice == "Random Forest":
        prediction = rf.predict(user_scaled)[0]
        confidence = np.max(rf.predict_proba(user_scaled)) * 100

    elif model_choice == "SVM":
        prediction = svm.predict(user_scaled)[0]
        confidence = np.max(svm.predict_proba(user_scaled)) * 100

    elif model_choice == "Logistic Regression":
        prediction = lr.predict(user_scaled)[0]
        confidence = np.max(lr.predict_proba(user_scaled)) * 100

    elif model_choice == "XGBoost":
        prediction_encoded = xgb.predict(user_scaled)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]
        confidence = np.max(xgb.predict_proba(user_scaled)) * 100

    # Save in session
    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.confidence = confidence

def sumgen(prediction):
    if prediction == "Independent":
        summary = "🌿 You are self-driven, confident and prefer working on your own goals. You value freedom, personal space and independent decision-making."
        rec = "You perform best in roles that allow autonomy and flexible thinking."
        hob = """
        <li>    Solo travel ✈️🌍
	    <li>	Freelancing 💻🧑‍💼
	    <li>	Blogging ✍️📝
	    <li>	Fitness training 🏋️‍♀️💪
"""
    elif prediction == "Analytical":
        summary = "📊 You are logical, detail-oriented and enjoy solving structured problems. You make decisions based on data rather than emotions."
        rec = "You thrive in environments that require research, strategy and critical thinking."
        hob = """
        <li>	Chess ♟️🧠
	    <li>	Coding challenges 💻👨‍💻
	    <li>	Stock market analysis 📈💹
	    <li>	Puzzle solving 🧩🔍
"""
    elif prediction == "Creative":
        summary = "🎨 You are imaginative, expressive and enjoy generating new ideas. You think differently and bring innovation to tasks."
        rec = "You excel in fields that encourage originality and design thinking."
        hob = """
        <li>	Photography 📸✨
	    <li>	Content creation 🎥📱
	    <li>	Fashion designing 👗🪡
	    <li>	Music / Painting 🎶🎨
"""
    elif prediction == "Leader":
        summary = "👑You are confident, decisive and naturally guide others. You take responsibility and motivate teams toward goals."
        rec = " You perform strongly in managerial and team-based roles."
        hob = """
        <li>	Event management 🎤📋
	    <li>	Debate club 🎙️🗣️
	    <li>	Student council 🏛️🗳️
	    <li>	Entrepreneurship 🚀💼
"""

    return summary, rec, hob
# ✅ SHOW RESULT AFTER BUTTON (OUTSIDE BUTTON BLOCK)

if st.session_state.get("prediction_done"):

    summary, rec, hob = sumgen(st.session_state.prediction)

    # Radar Data
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

    col1, col2 = st.columns([1,1])

    # LEFT COLUMN
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #E06AA1, #6E1A7A);
            padding: 25px;
            border-radius: 18px;
            color: white;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        ">
            <h2 text-align: center;>🎯Predicted Personality</h2>
            <h4 text-align: left;>{st.session_state.prediction}</h4>
            <p style="font-size:18px;">
                📊 Confidence Score: <b>{st.session_state.confidence:.2f}%</b>
            </p>
            <hr>
            <p>{summary}</p>
            <p><b>Recommendation:</b> {rec}</p>
            <p><b>Suggested Hobbies:</b><br>{hob}</p>
        </div>
            
        """, unsafe_allow_html=True)

    # RIGHT COLUMN 057C85
    with col2:
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(5,5))
        ax.plot(angles, values, color="#FF4B91", linewidth=3)
        ax.fill(angles, values, color="#FF4B91", alpha=0.3)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(traits, color="white")
        ax.set_yticklabels([])
        ax.spines['polar'].set_color('white')

        fig.patch.set_facecolor("#2D142C")
        ax.set_facecolor("#1E1E2E")

        st.pyplot(fig)



# --------------------------------------------------
# 📈 Feature Importance (Random Forest)
# --------------------------------------------------
st.subheader("📈 Feature Importance (Random Forest)")

importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.barh(features[indices], importances[indices], color="#C9005A", edgecolor = "white", linewidth=1.5)
ax2.set_title("Feature Importance (Random Forest)")
ax2.set_xlabel("Importance Score")
st.pyplot(fig2)


st.markdown("---")
st.caption("Advanced AI/ML Personality Classification Project | Premium Dashboard UI")

