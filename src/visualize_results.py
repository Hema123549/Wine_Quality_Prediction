import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv("outputs/prepared_wine.csv")

X = df.drop("quality_label", axis=1)
y = df["quality_label"]

# Same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load models
logistic_model = joblib.load("outputs/logistic_model.pkl")
rf_model = joblib.load("outputs/random_forest_model.pkl")

# Generate predictions
logistic_pred = logistic_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# 1. Model Accuracy Comparison
models = ["Logistic Regression", "Random Forest"]
accuracies = [
    logistic_model.score(X_test, y_test),
    rf_model.score(X_test, y_test)
]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("outputs/model_accuracy_comparison.png")
plt.show()

# 2. Random Forest Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Bad Quality", "Good Quality"],
    yticklabels=["Bad Quality", "Good Quality"]
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/random_forest_confusion_matrix.png")
plt.show()

# 3. Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, logistic_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_log,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Bad Quality", "Good Quality"],
    yticklabels=["Bad Quality", "Good Quality"]
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/logistic_confusion_matrix.png")
plt.show()

print("Model Visualization Completed Successfully!")