

import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


# ============================================================
# 1. Configuration
# ============================================================


from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "Electricity consumption in Finland"

TEMPERATURE = "Average temperature [°C]"

ALL_FEATURES = [
    TEMPERATURE,
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
]

FEATURES_WITHOUT_TEMPERATURE = [
    feature
    for feature in ALL_FEATURES
    if feature != TEMPERATURE
]


# ============================================================
# 2. Load data
# ============================================================

print("Loading processed datasets...")

train_data = pd.read_csv(
    os.path.join(DATA_DIR, "train_data.csv")
)

validation_data = pd.read_csv(
    os.path.join(DATA_DIR, "validation_data.csv")
)

test_data = pd.read_csv(
    os.path.join(DATA_DIR, "test_data.csv")
)

print("Train shape:", train_data.shape)
print("Validation shape:", validation_data.shape)
print("Test shape:", test_data.shape)


# ============================================================
# 3. Evaluation function
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)

    rmse = mean_squared_error(
        y,
        predictions
    ) ** 0.5

    return mae, rmse


# ============================================================
# 4. Model definitions
# ============================================================

def create_linear_regression():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])


def create_random_forest():

    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )


def create_xgboost():

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )


# ============================================================
# 5. Run experiment
# ============================================================

results = []

models = {
    "Linear Regression": create_linear_regression,
    "Random Forest": create_random_forest,
    "XGBoost": create_xgboost
}


for feature_set_name, feature_columns in [
    ("With Temperature", ALL_FEATURES),
    ("Without Temperature", FEATURES_WITHOUT_TEMPERATURE)
]:

    print("\n========================================")
    print(feature_set_name)
    print("========================================")

    X_train = train_data[feature_columns]
    y_train = train_data[TARGET]

    X_validation = validation_data[feature_columns]
    y_validation = validation_data[TARGET]

    X_test = test_data[feature_columns]
    y_test = test_data[TARGET]

    # Rename temperature for XGBoost compatibility
    xgb_features = [
        "temperature"
        if feature == TEMPERATURE
        else feature
        for feature in feature_columns
    ]

    X_train_xgb = X_train.copy()
    X_validation_xgb = X_validation.copy()
    X_test_xgb = X_test.copy()

    if TEMPERATURE in feature_columns:

        X_train_xgb = X_train_xgb.rename(
            columns={TEMPERATURE: "temperature"}
        )

        X_validation_xgb = X_validation_xgb.rename(
            columns={TEMPERATURE: "temperature"}
        )

        X_test_xgb = X_test_xgb.rename(
            columns={TEMPERATURE: "temperature"}
        )

    for model_name, model_function in models.items():

        print(f"\nTraining {model_name}...")

        model = model_function()

        if model_name == "XGBoost":

            model.fit(
                X_train_xgb,
                y_train
            )

            validation_mae, validation_rmse = evaluate_model(
                model,
                X_validation_xgb,
                y_validation
            )

            test_mae, test_rmse = evaluate_model(
                model,
                X_test_xgb,
                y_test
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            validation_mae, validation_rmse = evaluate_model(
                model,
                X_validation,
                y_validation
            )

            test_mae, test_rmse = evaluate_model(
                model,
                X_test,
                y_test
            )

        results.append({
            "Model": model_name,
            "Feature_Set": feature_set_name,
            "Validation_MAE": validation_mae,
            "Validation_RMSE": validation_rmse,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse
        })

        print(f"Validation MAE:  {validation_mae:.4f}")
        print(f"Validation RMSE: {validation_rmse:.4f}")
        print(f"Test MAE:        {test_mae:.4f}")
        print(f"Test RMSE:       {test_rmse:.4f}")


# ============================================================
# 6. Save results
# ============================================================

results_df = pd.DataFrame(results)

output_path = os.path.join(
    RESULTS_DIR,
    "temperature_ablation_results.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n========================================")
print("TEMPERATURE ABLATION RESULTS")
print("========================================")

print(results_df.to_string(index=False))

print("\nResults saved to:")
print(output_path)


# ============================================================
# 7. Calculate temperature contribution
# ============================================================

summary = []

for model_name in models.keys():

    with_temperature = results_df[
        (results_df["Model"] == model_name) &
        (results_df["Feature_Set"] == "With Temperature")
    ].iloc[0]

    without_temperature = results_df[
        (results_df["Model"] == model_name) &
        (results_df["Feature_Set"] == "Without Temperature")
    ].iloc[0]

    summary.append({
        "Model": model_name,

        "Test_MAE_With_Temperature":
            with_temperature["Test_MAE"],

        "Test_MAE_Without_Temperature":
            without_temperature["Test_MAE"],

        "MAE_Increase_After_Removal":
            without_temperature["Test_MAE"] -
            with_temperature["Test_MAE"],

        "Test_RMSE_With_Temperature":
            with_temperature["Test_RMSE"],

        "Test_RMSE_Without_Temperature":
            without_temperature["Test_RMSE"],

        "RMSE_Increase_After_Removal":
            without_temperature["Test_RMSE"] -
            with_temperature["Test_RMSE"]
    })


summary_df = pd.DataFrame(summary)

summary_path = os.path.join(
    RESULTS_DIR,
    "temperature_ablation_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n========================================")
print("TEMPERATURE CONTRIBUTION SUMMARY")
print("========================================")

print(summary_df.to_string(index=False))

print("\nSummary saved to:")
print(summary_path)

print("\nTemperature ablation experiment completed.")