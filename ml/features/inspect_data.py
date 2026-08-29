import pandas as pd

DATA_PATH = "data/raw/creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("First 5 rows: ")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nTime statistics:")
print(df["Time"].describe())

print("\nAmount statistics:")
print(df["Amount"].describe())