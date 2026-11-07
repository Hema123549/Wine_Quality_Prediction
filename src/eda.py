import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
red = pd.read_csv("dataset/winequality-red.csv", sep=";")
white = pd.read_csv("dataset/winequality-white.csv", sep=";")

# Add wine type
red['wine_type'] = 'red'
white['wine_type'] = 'white'

# Combine datasets
df = pd.concat([red, white], ignore_index=True)

# Save combined dataset
df.to_csv("outputs/combined_wine.csv", index=False)

sns.set_style("whitegrid")

# 1. Wine Quality Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='quality')
plt.title("Wine Quality Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.savefig("outputs/quality_distribution.png")
plt.show()

# 2. Alcohol vs Quality
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='quality', y='alcohol')
plt.title("Alcohol vs Wine Quality")
plt.savefig("outputs/alcohol_vs_quality.png")
plt.show()

# 3. Volatile Acidity vs Quality
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='quality', y='volatile acidity')
plt.title("Volatile Acidity vs Wine Quality")
plt.savefig("outputs/volatile_acidity_vs_quality.png")
plt.show()

# 4. Red vs White Wine Count
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='wine_type')
plt.title("Red Wine vs White Wine Count")
plt.savefig("outputs/wine_type_distribution.png")
plt.show()

# 5. Correlation Heatmap
df_encoded = pd.get_dummies(
    df,
    columns=['wine_type'],
    drop_first=True,
    dtype=int
)

plt.figure(figsize=(12, 8))
sns.heatmap(
    df_encoded.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Wine Quality Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/wine_correlation_heatmap.png")
plt.show()

print("EDA Completed Successfully!")