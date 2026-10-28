import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. File paths
# ============================================================


from pathlib import Path

# Project paths
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

print("Datasets loaded successfully.")


# ============================================================
# 3. Define features and target
# ============================================================

TARGET = "Electricity consumption in Finland"

FEATURE_COLUMNS = [
    "Average temperature [°C]",
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
# 5. Create Random Forest model
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 6. Train model
# ============================================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

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
# 10. Save metrics
# ============================================================

results = pd.DataFrame({
    "Model": ["Random Forest"],
    "Validation_MAE": [validation_mae],
    "Validation_RMSE": [validation_rmse],
    "Test_MAE": [test_mae],
    "Test_RMSE": [test_rmse]
})

results_path = os.path.join(
    RESULTS_DIR,
    "random_forest_results.csv"
)

results.to_csv(results_path, index=False)

print(f"\nResults saved to: {results_path}")


# ============================================================
# 11. Save predictions
# ============================================================

prediction_results = pd.DataFrame({
    "timestamp_utc": test_data["timestamp_utc"],
    "actual_demand": y_test.values,
    "predicted_demand": test_predictions
})

prediction_path = os.path.join(
    RESULTS_DIR,
    "random_forest_predictions.csv"
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
    "random_forest_feature_importance.csv"
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
# 13. Plot predictions
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
    "Random Forest: Actual vs Predicted Electricity Demand"
)

plt.xlabel("Test Time Steps")
plt.ylabel("Electricity Demand")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "random_forest_forecast_test.png"
)

plt.savefig(plot_path, dpi=300)

plt.show()

print(f"Plot saved to: {plot_path}")

print("\nRandom Forest execution completed successfully.")