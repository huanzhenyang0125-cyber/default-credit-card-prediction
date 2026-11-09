import pandas as pd
from pathlib import Path

source = Path(__file__).resolve().parent / "data/raw/default of credit card clients.xls"
data = pd.read_excel(source, header=1, engine="calamine")

print("First five customers:")
print(data.head().to_string(index=False))
print("\nRows and columns:", data.shape)
print("Missing values:", data.isna().sum().sum())
print("Duplicate customer IDs:", data["ID"].duplicated().sum())
print("\nNext-month default counts:")
print(data["default payment next month"].value_counts().sort_index())
print("Default rate:", f"{data['default payment next month'].mean():.2%}")
print("\nAge and credit limit summary:")
print(data[["AGE", "LIMIT_BAL"]].describe().to_string())
