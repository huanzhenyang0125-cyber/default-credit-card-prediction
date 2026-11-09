"""Data ingestion and verification script for the UCI credit card default dataset."""
from pathlib import Path
import pandas as pd


def inspect_raw_data() -> pd.DataFrame:
    """Read and display descriptive data quality summaries of the raw dataset."""
    source = Path(__file__).resolve().parent / "data/raw/default of credit card clients.xls"
    data = pd.read_excel(source, header=1, engine="calamine")

    print("First five customers:")
    print(data.head().to_string(index=False))
    print(f"\nRows and columns: {data.shape}")
    print(f"Missing values: {data.isna().sum().sum()}")
    print(f"Duplicate customer IDs: {data['ID'].duplicated().sum()}")
    print("\nNext-month default counts:")
    print(data["default payment next month"].value_counts().sort_index())
    print(f"Default rate: {data['default payment next month'].mean():.2%}")
    print("\nAge and credit limit summary:")
    print(data[["AGE", "LIMIT_BAL"]].describe().to_string())
    return data


if __name__ == "__main__":
    inspect_raw_data()
