import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# Ladda dataset
data = pd.read_csv("winequalityWhite.csv", sep=';')

X = data.drop("quality", axis=1)
y = data["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # IMPORTANT for imbalanced datasets
)

X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"  # IMPORTANT for imbalance
    ))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



f1 = f1_score(y_test, y_pred, average="weighted")
print("\nWeighted F1-score:", f1)

joblib.dump(pipeline, "original_model.pkl")
print("\nModel saved as original_model.pkl")