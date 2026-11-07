import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load prepared dataset
df = pd.read_csv("outputs/prepared_wine.csv")

# Separate features and target
X = df.drop("quality_label", axis=1)
y = df["quality_label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("TRAINING DATA SHAPE:", X_train.shape)
print("TESTING DATA SHAPE:", X_test.shape)

# Define Logistic Regression Pipeline
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])

# Train Logistic Regression
logistic_model.fit(X_train, y_train)

# Predictions
logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

# Define Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train Random Forest
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]


# Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["Bad Quality", "Good Quality"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# Evaluate both models
evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    logistic_pred,
    logistic_prob
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_pred,
    rf_prob
)

# Save trained models
joblib.dump(logistic_model, "outputs/logistic_model.pkl")
joblib.dump(rf_model, "outputs/random_forest_model.pkl")

print("\nBoth Models Saved Successfully!")