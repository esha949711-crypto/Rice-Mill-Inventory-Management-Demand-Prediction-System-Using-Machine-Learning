"""Train, compare, and save regression models for rice demand prediction."""

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import (
    TARGET,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_and_engineer,
    make_features,
)


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "rice_inventory.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

raw_data = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print(f"Total rows: {len(raw_data)}")
print(f"Total columns: {len(raw_data.columns)}")


# ============================================================
# CLEAN AND FEATURE ENGINEERING
# ============================================================

processed_data = clean_and_engineer(raw_data)

processed_data["date"] = pd.to_datetime(
    processed_data["date"],
    errors="coerce"
)

processed_data = processed_data.dropna(subset=["date"])

processed_data = (
    processed_data
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split = int(len(processed_data) * 0.80)

train = processed_data.iloc[:split]
test = processed_data.iloc[split:]

print("\nData Split")
print("-" * 50)
print(f"Training rows : {len(train)}")
print(f"Testing rows  : {len(test)}")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X_train = make_features(train)
X_test = make_features(test)

y_train = train[TARGET]
y_test = test[TARGET]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    [
        (
            "num",
            StandardScaler(),
            NUMERIC_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
        ),
    ]
)


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=180,
        max_depth=2,
        learning_rate=0.05,
        random_state=42
    ),
}


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

results = []

print("\nTraining models...")

for name, estimator in models.items():

    print(f"Training: {name}")

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = mean_squared_error(
        y_test,
        pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        pred
    )

    results.append(
        {
            "model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
        }
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

metrics = (
    pd.DataFrame(results)
    .sort_values(
        ["RMSE", "MAE"],
        ascending=True
    )
    .reset_index(drop=True)
)

best_name = metrics.iloc[0]["model"]


# ============================================================
# TRAIN BEST MODEL AGAIN
# ============================================================

best = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", models[best_name])
    ]
)

best.fit(X_train, y_train)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(
    best,
    MODEL_DIR / "rice_demand_model.joblib"
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

metrics.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_data = {
    "best_model": best_name,
    "train_rows": len(train),
    "test_rows": len(test),
    "test_period": (
        f"{test.date.min().date()} "
        f"to "
        f"{test.date.max().date()}"
    ),
    "metrics": results,
}

(
    MODEL_DIR / "metrics.json"
).write_text(
    json.dumps(
        metrics_data,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT BORDERED MODEL COMPARISON TABLE
# ============================================================

print("\n")

# Table widths
MODEL_WIDTH = 24
MAE_WIDTH = 12
RMSE_WIDTH = 12
R2_WIDTH = 12

# Border line
border = (
    "+"
    + "-" * MODEL_WIDTH
    + "+"
    + "-" * MAE_WIDTH
    + "+"
    + "-" * RMSE_WIDTH
    + "+"
    + "-" * R2_WIDTH
    + "+"
)

# Title
print("=" * 64)
print("             RICE DEMAND MODEL COMPARISON")
print("=" * 64)

# Top border
print(border)

# Header
print(
    f"| {'Model':<{MODEL_WIDTH - 2}}"
    f"| {'MAE':>{MAE_WIDTH - 2}}"
    f"| {'RMSE':>{RMSE_WIDTH - 2}}"
    f"| {'R2':>{R2_WIDTH - 2}} |"
)

# Header separator
print(border)

# Table rows
for row in metrics.itertuples(index=False):

    print(
        f"| {row.model:<{MODEL_WIDTH - 2}}"
        f"| {row.MAE:>{MAE_WIDTH - 2}.3f}"
        f"| {row.RMSE:>{RMSE_WIDTH - 2}.3f}"
        f"| {row.R2:>{R2_WIDTH - 2}.4f} |"
    )

# Bottom border
print(border)


# ============================================================
# PROJECT INFORMATION TABLE
# ============================================================

print("\n")

INFO_WIDTH = 64

info_border = "+" + "-" * INFO_WIDTH + "+"

print(info_border)

print(
    f"| {'PROJECT INFORMATION':^{INFO_WIDTH - 2}} |"
)

print(info_border)

print(
    f"| {'Best Model':<25}: "
    f"{best_name:<{INFO_WIDTH - 29}}|"
)

print(
    f"| {'Training Rows':<25}: "
    f"{len(train):<{INFO_WIDTH - 29}}|"
)

print(
    f"| {'Testing Rows':<25}: "
    f"{len(test):<{INFO_WIDTH - 29}}|"
)

test_period = (
    f"{test.date.min().date()} "
    f"to "
    f"{test.date.max().date()}"
)

print(
    f"| {'Test Period':<25}: "
    f"{test_period:<{INFO_WIDTH - 29}}|"
)

print(info_border)


# ============================================================
# SAVED FILES TABLE
# ============================================================

print("\n")

print(info_border)

print(
    f"| {'SAVED PROJECT FILES':^{INFO_WIDTH - 2}} |"
)

print(info_border)

print(
    f"| {'Trained Model':<25}: "
    f"{'models/rice_demand_model.joblib':<{INFO_WIDTH - 29}}|"
)

print(
    f"| {'Model Comparison':<25}: "
    f"{'models/model_comparison.csv':<{INFO_WIDTH - 29}}|"
)

print(
    f"| {'Metrics':<25}: "
    f"{'models/metrics.json':<{INFO_WIDTH - 29}}|"
)

print(info_border)

print("\nTraining completed successfully!")