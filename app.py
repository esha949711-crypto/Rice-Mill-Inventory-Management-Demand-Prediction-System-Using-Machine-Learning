from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
MODEL_PATH = ROOT / "models" / "rice_demand_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_model_and_metrics():
    model = None
    metrics = {}

    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None

    if METRICS_PATH.exists():
        try:
            metrics = json.loads(METRICS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            metrics = {}

    return model, metrics


model, metrics = load_model_and_metrics()


def parse_numeric_value(value, field_name):
    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc

    if amount < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return amount


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form = {"date": "2025-01-01", "rice_type": "Basmati", "sales_kg": "300", "production_kg": "250", "purchases_kg": "150", "current_stock_kg": "500", "price_per_kg": "95"}
    error = None
    if request.method == "POST":
        form = request.form.to_dict()
        try:
            date = pd.to_datetime(form["date"])
            if form.get("rice_type") not in {"Basmati", "Sona Masuri", "Ponni"}:
                raise ValueError("Select a valid rice type.")

            values = {
                "sales_kg": parse_numeric_value(form["sales_kg"], "Sales"),
                "production_kg": parse_numeric_value(form["production_kg"], "Production"),
                "purchases_kg": parse_numeric_value(form["purchases_kg"], "Purchases"),
                "current_stock_kg": parse_numeric_value(form["current_stock_kg"], "Current stock"),
                "price_per_kg": parse_numeric_value(form["price_per_kg"], "Price per kg"),
            }

            if model is None:
                raise RuntimeError("The demand model is not available yet. Please run python train_model.py first.")

            row = pd.DataFrame([
                {
                    "sales_kg": values["sales_kg"],
                    "production_kg": values["production_kg"],
                    "purchases_kg": values["purchases_kg"],
                    "current_stock_kg": values["current_stock_kg"],
                    "price_per_kg": values["price_per_kg"],
                    "month": date.month,
                    "quarter": date.quarter,
                    "year": date.year,
                    "rice_type": form["rice_type"],
                }
            ])

            predicted = max(0.0, float(model.predict(row)[0]))
            stock = values["current_stock_kg"]
            safety_stock = 0.20 * predicted
            reorder_qty = max(0.0, predicted + safety_stock - stock)
            if stock < predicted * 0.80:
                status, badge = "Shortage Risk", "danger"
                recommendation = f"Reorder approximately {reorder_qty:,.0f} kg before the next period."
            elif stock > predicted * 1.50:
                status, badge = "Excess Inventory", "warning"
                recommendation = "Pause or reduce purchases and prioritize selling existing stock."
            else:
                status, badge = "Healthy Stock", "success"
                recommendation = "Stock is within a practical range; monitor the next period."
            result = {
                "predicted": predicted,
                "stock": stock,
                "status": status,
                "badge": badge,
                "recommendation": recommendation,
                "reorder": reorder_qty,
                "coverage": (stock / predicted * 100) if predicted else 0,
            }
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            error = f"Please enter valid values for every field. ({exc})"
        except Exception as exc:  # pragma: no cover - defensive fallback
            error = f"Unable to generate a prediction. ({exc})"

    return render_template(
        "index.html",
        result=result,
        form=form,
        error=error,
        best_model=metrics.get("best_model", "trained model"),
        metrics=metrics.get("metrics", []),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
