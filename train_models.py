# train_models.py

import pandas as pd
import os
import joblib
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv("realistic_personality_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X, y, y_encoded, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Train Models
# --------------------------------------------------

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

# --------------------------------------------------
# Accuracy Check
# --------------------------------------------------
print("Model Accuracies:")
print("KNN:", accuracy_score(y_test, best_knn.predict(X_test_scaled)))
print("Random Forest:", accuracy_score(y_test, rf.predict(X_test_scaled)))
print("SVM:", accuracy_score(y_test, svm.predict(X_test_scaled)))
print("Logistic Regression:", accuracy_score(y_test, lr.predict(X_test_scaled)))
print("XGBoost:", accuracy_score(y_test_enc, xgb.predict(X_test_scaled)))

# --------------------------------------------------
# Save Models
# --------------------------------------------------
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

joblib.dump(best_knn, "saved_models/KNN.pkl")
joblib.dump(rf, "saved_models/Random_Forest.pkl")
joblib.dump(svm, "saved_models/SVM.pkl")
joblib.dump(lr, "saved_models/Logistic_Regression.pkl")
joblib.dump(xgb, "saved_models/XGBoost.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(le, "saved_models/label_encoder.pkl")

print("All models saved successfully!")
