import joblib
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split



print("it is time to steal the model")
API_URL = "http://localhost:5000/predict"
original_model = joblib.load("original_model.pkl")

NUM_QUERIES = 15000


# Feature ranges (approximate from Wine dataset)
feature_ranges = [
    (3.8, 14.2),      # fixed acidity
    (0.08, 1.1),     # volatile acidity
    (0, 1.66),     # citric acid
    (0.6, 65.8),      # residual sugar
    (0.009, 0.346),     # chlorides
    (2, 289),     # free sulfur dioxide
    (9, 440),     # total sulfur dioxide
    (0.98711, 1.03898), # density
    (2.72, 3.8),   # pH
    (0.22, 1.08),     # sulphates
    (8, 14.2)       # alcohol
]


stolen_X = []
stolen_y = []



for _ in range(NUM_QUERIES):

    sample = [
        np.random.uniform(low, high)
        for low, high in feature_ranges
    ]


    response = requests.post(API_URL, json={"features": sample})

    if response.status_code == 200:
        data = response.json()
        stolen_X.append(sample)
        stolen_y.append(data["prediction"])

    else:
        print("Error:", response.text)

print("Data collection complete.")




print("Now it is time to train the stolen model...")

X = np.array(stolen_X)
y = np.array(stolen_y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

stolen_model = RandomForestClassifier(n_estimators=200, random_state=42)
stolen_model.fit(X_train, y_train)

y_pred = stolen_model.predict(X_test)

print("\n===== Stolen Model Performance =====")
print(classification_report(y_test, y_pred))


f1 = f1_score(y_test, y_pred, average="weighted")
print("\nWeighted F1-score:", f1)


X_test_real = pd.read_csv("X_test.csv")
y_test_real = pd.read_csv("y_test.csv").values.ravel()

original_preds = original_model.predict(X_test_real)
stolen_preds = stolen_model.predict(X_test_real)

print("Agreement:", accuracy_score(original_preds, stolen_preds))