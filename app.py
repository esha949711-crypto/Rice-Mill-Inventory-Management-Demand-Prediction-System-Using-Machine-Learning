
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, render_template_string

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "rice_inventory.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "rice_demand_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"

# Change this if your CSV uses another target column.
TARGET_COLUMN = "future_demand"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL DATA
# ============================================================

model = None
metrics_data = {}
feature_columns = []
training_message = ""


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Please place your CSV file inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:

        raise ValueError(
            "The CSV file is empty."
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataset(df):

    df = df.copy()

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = df.dropna(how="all")

    # --------------------------------------------------------
    # Date feature engineering
    # --------------------------------------------------------

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year

        # Date itself is not passed directly to model
        df = df.drop(columns=["date"])


    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:

        # Try common target names
        possible_targets = [
            "future_demand",
            "demand",
            "demand_kg",
            "future_demand_kg",
            "sales_kg"
        ]

        found_target = None

        for column in possible_targets:

            if column in df.columns:

                found_target = column
                break

        if found_target is None:

            raise ValueError(
                f"Target column '{TARGET_COLUMN}' was not found.\n\n"
                f"Available columns:\n{list(df.columns)}"
            )

        target = found_target

    else:

        target = TARGET_COLUMN


    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    df = df.dropna(
        subset=[target]
    )

    if len(df) < 20:

        raise ValueError(
            "The dataset needs at least 20 valid rows "
            "for model training."
        )


    X = df.drop(
        columns=[target]
    )

    y = df[target]


    # --------------------------------------------------------
    # Remove columns that cannot be useful
    # --------------------------------------------------------

    # Remove unnamed CSV index columns
    unwanted = [

        column
        for column in X.columns
        if column.lower().startswith("unnamed")
    ]

    if unwanted:

        X = X.drop(
            columns=unwanted
        )


    return X, y, target


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():

    global model
    global metrics_data
    global feature_columns
    global training_message


    df = load_dataset()

    X, y, target = prepare_dataset(df)

    feature_columns = list(
        X.columns
    )


    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )

        ]

    )


    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )

        ]

    )


    transformers = []


    if numeric_features:

        transformers.append(

            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )

        )


    if categorical_features:

        transformers.append(

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )

        )


    preprocessor = ColumnTransformer(

        transformers=transformers

    )


    # --------------------------------------------------------
    # Time-aware split
    # --------------------------------------------------------

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree Regressor":
            DecisionTreeRegressor(
                max_depth=10,
                random_state=RANDOM_STATE
            ),

        "Random Forest Regressor":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

        "Gradient Boosting Regressor":
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )

    }


    results = []

    fitted_models = {}


    # --------------------------------------------------------
    # Train + Evaluate
    # --------------------------------------------------------

    for model_name, estimator in models.items():

        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                )

            ]

        )


        pipeline.fit(
            X_train,
            y_train
        )


        predictions = pipeline.predict(
            X_test
        )


        mae = mean_absolute_error(
            y_test,
            predictions
        )


        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )


        r2 = r2_score(
            y_test,
            predictions
        )


        results.append(

            {

                "model": model_name,

                "mae": round(
                    float(mae),
                    4
                ),

                "rmse": round(
                    float(rmse),
                    4
                ),

                "r2": round(
                    float(r2),
                    4
                )

            }

        )


        fitted_models[
            model_name
        ] = pipeline


    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Lowest RMSE
    best_row = results_df.loc[
        results_df["rmse"].idxmin()
    ]

    best_model_name = best_row[
        "model"
    ]

    model = fitted_models[
        best_model_name
    ]


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Save actual vs predicted
    # --------------------------------------------------------

    best_predictions = model.predict(
        X_test
    )


    prediction_df = pd.DataFrame(

        {

            "actual":
                y_test.values,

            "predicted":
                best_predictions

        }

    )


    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_data = {

        "best_model":
            best_model_name,

        "target":
            target,

        "training_rows":
            len(X_train),

        "testing_rows":
            len(X_test),

        "metrics":
            results

    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_data,
            file,
            indent=4
        )


    training_message = (

        f"Training completed successfully. "
        f"Best model: {best_model_name}"

    )


    return metrics_data


# ============================================================
# INITIAL TRAINING
# ============================================================

try:

    train_models()

except Exception as exc:

    training_message = (
        f"Training error: {exc}"
    )

    print(
        "\nTRAINING ERROR:\n",
        exc
    )


# ============================================================
# HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Rice Mill ML Dashboard
</title>


<script src=
"https://cdn.jsdelivr.net/npm/chart.js">
</script>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        #f4f7fb;

    color:
        #1e293b;

}


/* =====================================================
   NAVBAR
===================================================== */

.navbar {

    background:
        #123b2a;

    color:
        white;

    padding:
        18px 6%;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;

}


.logo {

    font-size:
        22px;

    font-weight:
        bold;

}


.navbar a {

    color:
        white;

    text-decoration:
        none;

    margin-left:
        20px;

}


/* =====================================================
   CONTAINER
===================================================== */

.container {

    width:
        92%;

    max-width:
        1250px;

    margin:
        30px auto;

}


/* =====================================================
   HERO
===================================================== */

.hero {

    background:
        white;

    padding:
        35px;

    border-radius:
        18px;

    margin-bottom:
        25px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.07);

}


.hero h1 {

    margin-top:
        0;

    color:
        #123b2a;

}


.hero p {

    color:
        #64748b;

    line-height:
        1.7;

}


.model-badge {

    display:
        inline-block;

    background:
        #e8f5ed;

    color:
        #166534;

    padding:
        12px 18px;

    border-radius:
        10px;

    font-weight:
        bold;

}


/* =====================================================
   CARDS
===================================================== */

.card {

    background:
        white;

    padding:
        28px;

    border-radius:
        18px;

    margin-bottom:
        25px;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,0.07);

}


.card h2 {

    margin-top:
        0;

    color:
        #123b2a;

}


/* =====================================================
   FORM
===================================================== */

.grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        18px;

}


label {

    display:
        block;

    font-weight:
        bold;

    margin-bottom:
        7px;

}


input,
select {

    width:
        100%;

    padding:
        12px;

    border:
        1px solid #cbd5e1;

    border-radius:
        8px;

    font-size:
        15px;

}


button {

    width:
        100%;

    padding:
        14px;

    border:
        none;

    border-radius:
        9px;

    margin-top:
        20px;

    background:
        #123b2a;

    color:
        white;

    font-size:
        16px;

    font-weight:
        bold;

    cursor:
        pointer;

}


button:hover {

    background:
        #1d5c40;

}


/* =====================================================
   RESULT CARDS
===================================================== */

.results {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        15px;

}


.stat {

    background:
        #f8fafc;

    padding:
        22px;

    border-radius:
        12px;

    text-align:
        center;

}


.stat h3 {

    margin:
        0;

    font-size:
        27px;

    color:
        #123b2a;

}


.stat p {

    color:
        #64748b;

}


/* =====================================================
   STATUS
===================================================== */

.status {

    padding:
        18px;

    border-radius:
        12px;

    margin-top:
        20px;

}


.success {

    background:
        #dcfce7;

    color:
        #166534;

}


.warning {

    background:
        #fef3c7;

    color:
        #92400e;

}


.danger {

    background:
        #fee2e2;

    color:
        #991b1b;

}


/* =====================================================
   TABLE
===================================================== */

.table-wrapper {

    overflow-x:
        auto;

}


table {

    width:
        100%;

    border-collapse:
        collapse;

}


th,
td {

    padding:
        15px;

    text-align:
        center;

    border-bottom:
        1px solid #e5e7eb;

}


th {

    background:
        #f1f5f9;

    color:
        #123b2a;

}


tr:hover {

    background:
        #f8fafc;

}


/* =====================================================
   CHARTS
===================================================== */

.chart-box {

    position:
        relative;

    height:
        400px;

}


/* =====================================================
   INFO
===================================================== */

.info {

    color:
        #64748b;

    line-height:
        1.7;

}


.error {

    background:
        #fee2e2;

    color:
        #991b1b;

    padding:
        18px;

    border-radius:
        10px;

    margin-bottom:
        20px;

}


/* =====================================================
   FOOTER
===================================================== */

.footer {

    text-align:
        center;

    padding:
        30px;

    color:
        #64748b;

}


/* =====================================================
   RESPONSIVE
===================================================== */

@media(max-width: 800px) {

    .grid {

        grid-template-columns:
            1fr;

    }


    .results {

        grid-template-columns:
            repeat(2, 1fr);

    }


    .navbar {

        flex-direction:
            column;

        gap:
            12px;

    }

}


@media(max-width: 500px) {

    .results {

        grid-template-columns:
            1fr;

    }

}

</style>

</head>


<body>


<!-- =====================================================
     NAVBAR
===================================================== -->

<nav class="navbar">

    <div class="logo">

        🌾 Rice Mill ML System

    </div>


    <div>

        <a href="#prediction">
            Prediction
        </a>

        <a href="#models">
            Models
        </a>

        <a href="#graphs">
            Graphs
        </a>

    </div>

</nav>


<div class="container">


<!-- =====================================================
     HERO
===================================================== -->

<div class="hero">

    <h1>

        Rice Mill Inventory Management
        & Demand Prediction System

    </h1>


    <p>

        A Machine Learning based application
        that predicts future rice demand,
        evaluates multiple regression algorithms,
        and provides inventory management
        recommendations.

    </p>


    <div class="model-badge">

        🏆 Best Model:
        {{ best_model }}

    </div>

</div>


<!-- =====================================================
     ERROR / TRAINING MESSAGE
===================================================== -->

{% if error %}

<div class="error">

    {{ error }}

</div>

{% endif %}


<!-- =====================================================
     PREDICTION
===================================================== -->

<div class="card"
     id="prediction">

    <h2>
        🔮 Demand Prediction
    </h2>


    <p class="info">

        Enter rice mill information below.
        The selected Machine Learning model
        will predict future demand.

    </p>


    <form method="POST">


        <div class="grid">


            <div>

                <label>
                    Date
                </label>

                <input
                    type="date"
                    name="date"
                    value="{{ form.date }}"
                    required
                >

            </div>


            <div>

                <label>
                    Rice Type
                </label>

                <select
                    name="rice_type"
                    required
                >

                    <option>Basmati</option>

                    <option>Sona Masuri</option>

                    <option>Ponni</option>

                </select>

            </div>


            <div>

                <label>
                    Sales (kg)
                </label>

                <input
                    type="number"
                    step="0.01"
                    name="sales_kg"
                    value="{{ form.sales_kg }}"
                    required
                >

            </div>


            <div>

                <label>
                    Production (kg)
                </label>

                <input
                    type="number"
                    step="0.01"
                    name="production_kg"
                    value="{{ form.production_kg }}"
                    required
                >

            </div>


            <div>

                <label>
                    Purchases (kg)
                </label>

                <input
                    type="number"
                    step="0.01"
                    name="purchases_kg"
                    value="{{ form.purchases_kg }}"
                    required
                >

            </div>


            <div>

                <label>
                    Current Stock (kg)
                </label>

                <input
                    type="number"
                    step="0.01"
                    name="current_stock_kg"
                    value="{{ form.current_stock_kg }}"
                    required
                >

            </div>


            <div>

                <label>
                    Price per kg
                </label>

                <input
                    type="number"
                    step="0.01"
                    name="price_per_kg"
                    value="{{ form.price_per_kg }}"
                    required
                >

            </div>


        </div>


        <button type="submit">

            Predict Demand & Analyze Inventory

        </button>


    </form>

</div>


<!-- =====================================================
     PREDICTION RESULT
===================================================== -->

{% if result %}

<div class="card">

    <h2>
        📦 Inventory Prediction
    </h2>


    <div class="results">


        <div class="stat">

            <h3>

                {{ "%.0f"|format(result.predicted) }}

            </h3>

            <p>
                Predicted Demand (kg)
            </p>

        </div>


        <div class="stat">

            <h3>

                {{ "%.0f"|format(result.stock) }}

            </h3>

            <p>
                Current Stock (kg)
            </p>

        </div>


        <div class="stat">

            <h3>

                {{ "%.0f"|format(result.required_stock) }}

            </h3>

            <p>
                Required Stock (kg)
            </p>

        </div>


        <div class="stat">

            <h3>

                {{ "%.0f"|format(result.reorder) }}

            </h3>

            <p>
                Recommended Reorder (kg)
            </p>

        </div>


    </div>


    <div class="status {{ result.badge }}">

        <h3>

            Inventory Status:
            {{ result.status }}

        </h3>


        <p>

            {{ result.status_description }}

        </p>


        <p>

            <strong>
                Recommendation:
            </strong>

            {{ result.recommendation }}

        </p>


        <p>

            <strong>
                Stock Coverage:
            </strong>

            {{ "%.1f"|format(result.coverage) }}%

        </p>

    </div>

</div>

{% endif %}


<!-- =====================================================
     MODEL COMPARISON
===================================================== -->

<div class="card"
     id="models">


    <h2>

        🤖 Regression Model Comparison

    </h2>


    <p class="info">

        Four regression algorithms are trained
        and evaluated using the same test data.
        The model with the lowest RMSE is selected
        automatically.

    </p>


    <div class="table-wrapper">


        <table>


            <thead>

                <tr>

                    <th>
                        Model
                    </th>

                    <th>
                        MAE
                    </th>

                    <th>
                        RMSE
                    </th>

                    <th>
                        R² Score
                    </th>

                </tr>

            </thead>


            <tbody>


            {% for row in metrics %}

                <tr>

                    <td>

                        <strong>

                            {{ row.model }}

                        </strong>

                    </td>


                    <td>

                        {{ "%.2f"|format(row.mae) }}

                    </td>


                    <td>

                        {{ "%.2f"|format(row.rmse) }}

                    </td>


                    <td>

                        {{ "%.3f"|format(row.r2) }}

                    </td>

                </tr>

            {% endfor %}


            </tbody>

        </table>

    </div>


</div>


<!-- =====================================================
     CHARTS
===================================================== -->

<div class="card"
     id="graphs">


    <h2>

        📈 Model Error Comparison

    </h2>


    <p class="info">

        Lower MAE and RMSE indicate smaller
        prediction errors.

    </p>


    <div class="chart-box">

        <canvas id="errorChart">
        </canvas>

    </div>


</div>


<div class="card">


    <h2>

        📊 R² Score Comparison

    </h2>


    <p class="info">

        Higher R² generally indicates that the
        model explains more variation in the
        target variable.

    </p>


    <div class="chart-box">

        <canvas id="r2Chart">
        </canvas>

    </div>


</div>


<div class="card">


    <h2>

        🎯 Machine Learning Workflow

    </h2>


    <p class="info">

        CSV Dataset
        →
        Data Preprocessing
        →
        Feature Engineering
        →
        Train/Test Split
        →
        Linear Regression
        →
        Decision Tree
        →
        Random Forest
        →
        Gradient Boosting
        →
        Evaluation
        →
        Best Model
        →
        Demand Prediction
        →
        Inventory Recommendation

    </p>


</div>


</div>


<!-- =====================================================
     FOOTER
===================================================== -->

<div class="footer">

    Rice Mill Inventory Management & Demand Prediction System

    <br><br>

    Machine Learning • Flask • Scikit-learn

</div>


<!-- =====================================================
     CHART JAVASCRIPT
===================================================== -->

<script>


const models = [

{% for row in metrics %}

    "{{ row.model }}",

{% endfor %}

];


const mae = [

{% for row in metrics %}

    {{ row.mae }},

{% endfor %}

];


const rmse = [

{% for row in metrics %}

    {{ row.rmse }},

{% endfor %}

];


const r2 = [

{% for row in metrics %}

    {{ row.r2 }},

{% endfor %}

];


// ========================================================
// MAE + RMSE
// ========================================================

new Chart(

    document.getElementById(
        "errorChart"
    ),

    {

        type: "bar",

        data: {

            labels: models,

            datasets: [

                {

                    label: "MAE",

                    data: mae

                },

                {

                    label: "RMSE",

                    data: rmse

                }

            ]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {

                    position: "top"

                }

            },

            scales: {

                y: {

                    beginAtZero: true

                }

            }

        }

    }

);


// ========================================================
// R²
// ========================================================

new Chart(

    document.getElementById(
        "r2Chart"
    ),

    {

        type: "bar",

        data: {

            labels: models,

            datasets: [

                {

                    label: "R² Score",

                    data: r2

                }

            ]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {

                    position: "top"

                }

            },

            scales: {

                y: {

                    beginAtZero: true

                }

            }

        }

    }

);

</script>


</body>

</html>

"""


# ============================================================
# FLASK ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    result = None
    error = None


    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    form = {

        "date":
            "2025-01-01",

        "rice_type":
            "Basmati",

        "sales_kg":
            "300",

        "production_kg":
            "250",

        "purchases_kg":
            "150",

        "current_stock_kg":
            "500",

        "price_per_kg":
            "95"

    }


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if request.method == "POST":

        form = request.form.to_dict()

        try:

            if model is None:

                raise RuntimeError(
                    "No trained model is available. "
                    "Check your CSV and restart the application."
                )


            date = pd.to_datetime(
                form["date"]
            )


            def get_number(
                name
            ):

                value = float(
                    form[name]
                )

                if value < 0:

                    raise ValueError(
                        f"{name} cannot be negative."
                    )

                return value


            sales = get_number(
                "sales_kg"
            )

            production = get_number(
                "production_kg"
            )

            purchases = get_number(
                "purchases_kg"
            )

            current_stock = get_number(
                "current_stock_kg"
            )

            price = get_number(
                "price_per_kg"
            )


            rice_type = form[
                "rice_type"
            ]


            # ------------------------------------------------
            # Input dataframe
            # ------------------------------------------------

            input_data = pd.DataFrame(

                [

                    {

                        "sales_kg":
                            sales,

                        "production_kg":
                            production,

                        "purchases_kg":
                            purchases,

                        "current_stock_kg":
                            current_stock,

                        "price_per_kg":
                            price,

                        "month":
                            date.month,

                        "quarter":
                            date.quarter,

                        "year":
                            date.year,

                        "rice_type":
                            rice_type

                    }

                ]

            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predicted = model.predict(
                input_data
            )[0]


            predicted = max(
                0,
                float(predicted)
            )


            # ------------------------------------------------
            # Inventory calculations
            # ------------------------------------------------

            safety_stock = (
                predicted * 0.20
            )


            required_stock = (
                predicted +
                safety_stock
            )


            reorder_quantity = max(

                0,

                required_stock -
                current_stock

            )


            # ------------------------------------------------
            # Classification
            # ------------------------------------------------

            if predicted <= 0:

                status = "No Demand"

                badge = "warning"

                description = (
                    "The model predicts no significant "
                    "demand for the selected inputs."
                )


            elif current_stock < (
                predicted * 0.80
            ):

                status = "Shortage Risk"

                badge = "danger"

                description = (
                    "Current stock may be insufficient "
                    "to satisfy expected demand."
                )


            elif current_stock > (
                predicted * 1.50
            ):

                status = "Excess Inventory"

                badge = "warning"

                description = (
                    "Current inventory is considerably "
                    "higher than expected demand."
                )


            else:

                status = "Healthy Stock"

                badge = "success"

                description = (
                    "Current stock is within a practical "
                    "range compared with predicted demand."
                )


            # ------------------------------------------------
            # Recommendation
            # ------------------------------------------------

            if status == "Shortage Risk":

                recommendation = (
                    f"Reorder approximately "
                    f"{reorder_quantity:,.0f} kg."
                )

            elif status == "Excess Inventory":

                recommendation = (
                    "Reduce or pause new purchases "
                    "and prioritize existing inventory."
                )

            elif status == "No Demand":

                recommendation = (
                    "Monitor future demand before "
                    "placing additional purchases."
                )

            else:

                recommendation = (
                    "Continue normal inventory monitoring."
                )


            # ------------------------------------------------
            # Coverage
            # ------------------------------------------------

            coverage = (

                current_stock /
                predicted *
                100

                if predicted > 0

                else 0

            )


            result = {

                "predicted":
                    predicted,

                "stock":
                    current_stock,

                "safety_stock":
                    safety_stock,

                "required_stock":
                    required_stock,

                "reorder":
                    reorder_quantity,

                "status":
                    status,

                "badge":
                    badge,

                "status_description":
                    description,

                "recommendation":
                    recommendation,

                "coverage":
                    coverage

            }


        except Exception as exc:

            error = str(exc)


    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    return render_template_string(

        HTML,

        result=result,

        form=form,

        error=error,

        metrics=metrics_data.get(
            "metrics",
            []
        ),

        best_model=metrics_data.get(
            "best_model",
            "Not available"
        )

    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("🌾 RICE MILL ML INVENTORY SYSTEM")
    print("=" * 65)
    print(training_message)
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()
    print("Network:")
    print("http://192.168.100.60:5000")
    print("=" * 65)

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )
