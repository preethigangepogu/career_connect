import pandas as pd, numpy as np, joblib, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
n = 300
edu = ["High School", "Bachelor's", "Master's", "PhD"]
data = {
    "skills_match_score":     np.random.randint(20, 100, n),
    "years_of_experience":    np.random.randint(0, 15, n),
    "education_level":        np.random.choice(edu, n),
    "role_relevance_score":   np.random.randint(10, 100, n),
    "past_job_history_score": np.random.randint(10, 100, n),
}
df = pd.DataFrame(data)
def get_label(row):
    s = (row["skills_match_score"]*0.35 + row["role_relevance_score"]*0.25 +
         row["past_job_history_score"]*0.20 + min(row["years_of_experience"]*5,30)*0.20)
    return "Strong Match" if s>=70 else ("Moderate Match" if s>=45 else "Weak Match")
df["match_label"] = df.apply(get_label, axis=1)
le1, le2 = LabelEncoder(), LabelEncoder()
df["edu_enc"] = le1.fit_transform(df["education_level"])
df["lbl_enc"] = le2.fit_transform(df["match_label"])
X = df[["skills_match_score","years_of_experience","edu_enc","role_relevance_score","past_job_history_score"]]
y = df["lbl_enc"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
m = RandomForestClassifier(n_estimators=100, random_state=42)
m.fit(X_train, y_train)
acc = accuracy_score(y_test, m.predict(X_test))
print(f"Model accuracy: {round(acc*100,2)}%")
joblib.dump(m,   os.path.join(BASE, "rf_model.pkl"))
joblib.dump(le1, os.path.join(BASE, "edu_encoder.pkl"))
joblib.dump(le2, os.path.join(BASE, "label_encoder.pkl"))
print("Model files saved.")
