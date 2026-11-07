import pandas as pd
import joblib

# Load trained Random Forest model
model = joblib.load("outputs/random_forest_model.pkl")

# Load prepared dataset
df = pd.read_csv("outputs/prepared_wine.csv")

# Separate features and target
X = df.drop("quality_label", axis=1)
y = df["quality_label"]

# Select one actual good-quality wine sample
sample = X[y == 1].iloc[[0]]

# Predict wine quality
prediction = model.predict(sample)
probability = model.predict_proba(sample)[0]

print("WINE QUALITY PREDICTION")
print("-----------------------")

if prediction[0] == 1:
    print("Predicted Quality: Good Quality")
else:
    print("Predicted Quality: Bad Quality")

print(f"Probability of Bad Quality: {probability[0] * 100:.2f}%")
print(f"Probability of Good Quality: {probability[1] * 100:.2f}%")