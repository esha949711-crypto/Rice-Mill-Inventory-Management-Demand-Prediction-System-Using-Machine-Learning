import pandas as pd

from src.preprocessing import clean_and_engineer


def test_clean_and_engineer_removes_invalid_rows_and_creates_date_features():
    df = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "rice_type": "Basmati",
                "sales_kg": 100,
                "production_kg": 200,
                "purchases_kg": 50,
                "current_stock_kg": 150,
                "price_per_kg": 90,
                "future_demand_kg": 300,
            },
            {
                "date": "bad-date",
                "rice_type": "Ponni",
                "sales_kg": 50,
                "production_kg": 100,
                "purchases_kg": 40,
                "current_stock_kg": 80,
                "price_per_kg": 60,
                "future_demand_kg": 200,
            },
            {
                "date": "2024-02-01",
                "rice_type": "Sona Masuri",
                "sales_kg": "invalid",
                "production_kg": 180,
                "purchases_kg": 70,
                "current_stock_kg": 120,
                "price_per_kg": 65,
                "future_demand_kg": 240,
            },
        ]
    )

    result = clean_and_engineer(df)

    assert len(result) == 1
    assert result.iloc[0]["month"] == 1
    assert result.iloc[0]["quarter"] == 1
    assert result.iloc[0]["year"] == 2024
