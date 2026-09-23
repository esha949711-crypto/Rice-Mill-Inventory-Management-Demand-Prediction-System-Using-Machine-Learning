"""Feature preparation shared by model training and Flask inference."""
import pandas as pd

TARGET = "future_demand_kg"
NUMERIC_FEATURES = [
    "sales_kg",
    "production_kg",
    "purchases_kg",
    "current_stock_kg",
    "price_per_kg",
    "month",
    "quarter",
    "year",
]
CATEGORICAL_FEATURES = ["rice_type"]
FEATURE_COLUMNS = ["date", "rice_type", *NUMERIC_FEATURES[:5]]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date", "rice_type", TARGET]).copy()

    for col in [
        "sales_kg",
        "production_kg",
        "purchases_kg",
        "current_stock_kg",
        "price_per_kg",
        TARGET,
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[
        "sales_kg",
        "production_kg",
        "purchases_kg",
        "current_stock_kg",
        "price_per_kg",
        TARGET,
    ]).copy()

    data["month"] = data["date"].dt.month
    data["quarter"] = data["date"].dt.quarter
    data["year"] = data["date"].dt.year
    return data


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    if not {"month", "quarter", "year"}.issubset(df.columns):
        data = clean_and_engineer(df)
    else:
        data = df.copy()

    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
