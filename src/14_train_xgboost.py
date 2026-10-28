

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. File paths
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_PATH = PROCESSED_DIR / "train_data.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation_data.csv"
TEST_PATH = PROCESSED_DIR / "test_data.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Load datasets
# ============================================================

train_data = pd.read_csv(TRAIN_PATH)
validation_data = pd.read_csv(VALIDATION_PATH)
test_data = pd.read_csv(TEST_PATH)
# Rename columns to XGBoost-compatible feature names
SAFE_COLUMN_NAMES = {
    "Average temperature [°C]": "temperature",
    "Electricity consumption in Finland": "electricity_demand"
}

train_data = train_data.rename(columns=SAFE_COLUMN_NAMES)
validation_data = validation_data.rename(columns=SAFE_COLUMN_NAMES)
test_data = test_data.rename(columns=SAFE_COLUMN_NAMES)

print("Datasets loaded successfully.")

print("Training data shape:", train_data.shape)
print("Validation data shape:", validation_data.shape)
print("Test data shape:", test_data.shape)


# ============================================================
# 3. Define features and target
# ============================================================

TARGET = "electricity_demand"

FEATURE_COLUMNS = [
    "temperature",
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
    "rolling_mean_168"
]


# ============================================================
# 4. Prepare data
# ============================================================

X_train = train_data[FEATURE_COLUMNS]
y_train = train_data[TARGET]

X_validation = validation_data[FEATURE_COLUMNS]
y_validation = validation_data[TARGET]

X_test = test_data[FEATURE_COLUMNS]
y_test = test_data[TARGET]


# ============================================================
# 5. Create XGBoost model
# ============================================================

model = XGBRegressor(
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
# 6. Train model
# ============================================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_validation, y_validation)],
    verbose=False
)

print("Training completed.")


# ============================================================
# 7. Generate predictions
# ============================================================

validation_predictions = model.predict(X_validation)
test_predictions = model.predict(X_test)


# ============================================================
# 8. Evaluation function
# ============================================================

def evaluate_model(actual_values, predicted_values, dataset_name):

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    print(f"\n{dataset_name} Results")
    print("-" * 40)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mae, rmse


# ============================================================
# 9. Evaluate model
# ============================================================

validation_mae, validation_rmse = evaluate_model(
    y_validation,
    validation_predictions,
    "Validation"
)

test_mae, test_rmse = evaluate_model(
    y_test,
    test_predictions,
    "Test"
)


# ============================================================
# 10. Save evaluation results
# ============================================================

results = pd.DataFrame({
    "Model": ["XGBoost"],
    "Validation_MAE": [validation_mae],
    "Validation_RMSE": [validation_rmse],
    "Test_MAE": [test_mae],
    "Test_RMSE": [test_rmse]
})

results_path = os.path.join(
    RESULTS_DIR,
    "xgboost_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print(f"\nResults saved to: {results_path}")


# ============================================================
# 11. Save test predictions
# ============================================================

prediction_results = pd.DataFrame({
    "timestamp_utc": test_data["timestamp_utc"],
    "actual_demand": y_test.values,
    "predicted_demand": test_predictions
})

prediction_path = os.path.join(
    RESULTS_DIR,
    "xgboost_predictions.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)

print(f"Predictions saved to: {prediction_path}")


# ============================================================
# 12. Feature importance
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

importance_path = os.path.join(
    RESULTS_DIR,
    "xgboost_feature_importance.csv"
)

feature_importance.to_csv(
    importance_path,
    index=False
)

print("\nFeature importance:")
print(feature_importance)

print(
    f"\nFeature importance saved to: {importance_path}"
)


# ============================================================
# 13. Plot actual vs predicted demand
# ============================================================

plot_points = min(168, len(y_test))

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.iloc[:plot_points].values,
    label="Actual Demand"
)

plt.plot(
    test_predictions[:plot_points],
    label="Predicted Demand"
)

plt.title(
    "XGBoost: Actual vs Predicted Electricity Demand"
)

plt.xlabel("Test Time Steps")
plt.ylabel("Electricity Demand")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "xgboost_forecast_test.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()

print(f"Plot saved to: {plot_path}")

print("\nXGBoost execution completed successfully.")