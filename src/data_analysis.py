import pandas as pd

# Load Red Wine Dataset
red = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Load White Wine Dataset
white = pd.read_csv("dataset/winequality-white.csv", sep=";")

# Analyze Red Wine
print("========== RED WINE DATASET ==========")

print("\nFIRST FIVE ROWS:")
print(red.head())

print("\nDATASET SHAPE:")
print(red.shape)

print("\nMISSING VALUES:")
print(red.isnull().sum())

print("\nQUALITY COUNTS:")
print(red['quality'].value_counts().sort_index())

# Analyze White Wine
print("\n========== WHITE WINE DATASET ==========")

print("\nFIRST FIVE ROWS:")
print(white.head())

print("\nDATASET SHAPE:")
print(white.shape)

print("\nMISSING VALUES:")
print(white.isnull().sum())

print("\nQUALITY COUNTS:")
print(white['quality'].value_counts().sort_index())

# Combine both datasets
red['wine_type'] = 'red'
white['wine_type'] = 'white'

df = pd.concat([red, white], ignore_index=True)

print("\n========== COMBINED DATASET ==========")

print("\nCOMBINED SHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nWINE TYPE COUNTS:")
print(df['wine_type'].value_counts())

print("\nQUALITY COUNTS:")
print(df['quality'].value_counts().sort_index())

print("\nDATA ANALYSIS COMPLETED!")