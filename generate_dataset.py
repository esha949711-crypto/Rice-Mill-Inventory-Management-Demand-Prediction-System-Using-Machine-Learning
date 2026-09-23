"""Generate the deterministic educational rice inventory dataset.

Run from the project root with: python data/generate_dataset.py
"""
from pathlib import Path
import csv
import math
import random

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "rice_inventory.csv"
RANDOM_SEED = 42


def build_rows():
    random.seed(RANDOM_SEED)
    rice_types = {
        "Basmati": {"base": 330, "price": 92},
        "Sona Masuri": {"base": 270, "price": 64},
        "Ponni": {"base": 220, "price": 58},
    }
    rows = []
    months = [(year, month) for year in range(2022, 2025) for month in range(1, 13)]

    for index, (year, month) in enumerate(months):
        seasonal = 1 + 0.16 * math.sin((month - 2) * 2 * math.pi / 12)
        seasonal += 0.08 if month in (10, 11, 12) else 0
        trend = 1 + index * 0.006

        for rice_type, params in rice_types.items():
            type_factor = {"Basmati": 1.08, "Sona Masuri": 1.0, "Ponni": 0.94}[rice_type]
            demand = params["base"] * seasonal * trend * type_factor + random.gauss(0, 14)
            sales = max(80, round(demand * (0.94 + random.random() * 0.08)))
            production = max(90, round(demand * (0.72 + random.random() * 0.34)))
            purchases = max(50, round(demand * (0.38 + random.random() * 0.34)))
            opening_stock = params["base"] * (1.25 + random.random() * 0.65)
            current_stock = max(40, round(opening_stock + production + purchases - sales))
            price = round(params["price"] * (1 + 0.012 * index + random.gauss(0, 0.018)), 2)
            rows.append({
                "date": f"{year}-{month:02d}-01",
                "rice_type": rice_type,
                "sales_kg": sales,
                "production_kg": production,
                "purchases_kg": purchases,
                "current_stock_kg": current_stock,
                "price_per_kg": price,
                "future_demand_kg": round(max(70, demand), 2),
            })
    return rows


def main():
    rows = build_rows()
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
