"""
ML Model Training - Candidate Job Fit Prediction
Algorithm: Random Forest Classifier
Author: [Your Name]
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# ─────────────────────────────────────────────
# 1. Generate / Load Dataset
# ─────────────────────────────────────────────
np.random.seed(42)
n = 300

education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
roles = ["Frontend Developer", "Backend Developer", "Data Scientist",
         "DevOps Engineer", "UI/UX Designer", "ML Engineer"]

data = {
    "skills_match_score": np.random.randint(20, 100, n),
    "years_of_experience": np.random.randint(0, 15, n),
    "education_level": np.random.choice(education_levels, n),
    "role_relevance_score": np.random.randint(10, 100, n),
    "past_job_history_score": np.random.randint(10, 100, n),
}

df = pd.DataFrame(data)

# Create target label based on scores (simulate realistic labeling)
def assign_label(row):
    total = (row["skills_match_score"] * 0.35 +
             row["role_relevance_score"] * 0.25 +
             row["past_job_history_score"] * 0.20 +
             min(row["years_of_experience"] * 5, 30) * 0.20)
    if total >= 70:
        return "Strong Match"
    elif total >= 45:
        return "Moderate Match"
    else:
        return "Weak Match"

df["match_label"] = df.apply(assign_label, axis=1)

# Save dataset
df.to_csv("dataset.csv", index=False)
print("Dataset created. Shape:", df.shape)
print(df["match_label"].value_counts())

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
le_edu = LabelEncoder()
df["education_encoded"] = le_edu.fit_transform(df["education_level"])

le_label = LabelEncoder()
df["label_encoded"] = le_label.fit_transform(df["match_label"])

features = ["skills_match_score", "years_of_experience",
            "education_encoded", "role_relevance_score", "past_job_history_score"]

X = df[features]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
# 3. Train Random Forest
# ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
present_labels = sorted(set(y_test) | set(y_pred))
present_names = le_label.inverse_transform(present_labels)
print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_names))

# ─────────────────────────────────────────────
# 5. Save Model & Encoders
# ─────────────────────────────────────────────
joblib.dump(model, "rf_model.pkl")
joblib.dump(le_edu, "edu_encoder.pkl")
joblib.dump(le_label, "label_encoder.pkl")
print("\nModel saved: rf_model.pkl")
print("Encoders saved: edu_encoder.pkl, label_encoder.pkl")
print("\nDone! You can now run the backend.")
