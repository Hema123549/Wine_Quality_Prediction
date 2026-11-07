import pandas as pd

# Load combined dataset
df = pd.read_csv("outputs/combined_wine.csv")

# Convert quality into Good and Bad categories
df['quality_label'] = df['quality'].apply(
    lambda x: 1 if x >= 7 else 0
)

# Remove original quality column
df = df.drop('quality', axis=1)

# Convert wine_type into numerical format
df = pd.get_dummies(
    df,
    columns=['wine_type'],
    drop_first=True,
    dtype=int
)

# Display prepared dataset
print("PREPARED DATASET:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nQUALITY CATEGORY COUNTS:")
print(df['quality_label'].value_counts())

print("\nQUALITY CATEGORY PERCENTAGES:")
print(df['quality_label'].value_counts(normalize=True) * 100)

# Save prepared dataset
df.to_csv("outputs/prepared_wine.csv", index=False)

print("\nData Preparation Completed Successfully!")