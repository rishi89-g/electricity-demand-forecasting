
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

target = "Electricity consumption in Finland"
temperature_col = "Average temperature [°C]"

features = [
    temperature_col,
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

scenario_temperatures = [-15, 0, 10]
temperature_grid = np.arange(-15, 11, 1)

print("Loading data...")

train_data = pd.read_csv(PROCESSED_DIR / "train_data.csv")
test_data = pd.read_csv(PROCESSED_DIR / "test_data.csv")

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

print("Training models...")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train, y_train)

X_train_xgb = X_train.rename(
    columns={temperature_col: "temperature"}
)
X_test_xgb = X_test.rename(
    columns={temperature_col: "temperature"}
)

xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(
    X_train_xgb,
    y_train,
    eval_set=[(X_test_xgb, y_test)],
    verbose=False,
)

lstm_model = load_model(
    RESULTS_DIR / "lstm_model_with_temperature.keras"
)
feature_scaler = joblib.load(
    RESULTS_DIR / "lstm_feature_scaler_with_temperature.joblib"
)
target_scaler = joblib.load(
    RESULTS_DIR / "lstm_target_scaler_with_temperature.joblib"
)

results = []
prediction_curves = {}

for temperature in temperature_grid:
    X_scenario = X_test.copy()
    X_scenario[temperature_col] = temperature

    predictions = {
        "Linear Regression": lr_model.predict(X_scenario),
        "Random Forest": rf_model.predict(X_scenario),
        "XGBoost": xgb_model.predict(
            X_scenario.rename(
                columns={temperature_col: "temperature"}
            )
        ),
    }

    scaled_features = feature_scaler.transform(
        X_scenario[features].to_numpy()
    )

    sequences = np.stack([
        scaled_features[i - 23:i + 1]
        for i in range(23, len(scaled_features))
    ])

    lstm_scaled_predictions = lstm_model.predict(
        sequences,
        verbose=0,
    )

    predictions["LSTM"] = target_scaler.inverse_transform(
        lstm_scaled_predictions.reshape(-1, 1)
    ).ravel()

    for model_name, values in predictions.items():
        results.append({
            "Temperature_C": temperature,
            "Model": model_name,
            "Mean_Predicted_Demand": float(np.mean(values)),
            "Min_Predicted_Demand": float(np.min(values)),
            "Max_Predicted_Demand": float(np.max(values)),
        })

    if temperature in scenario_temperatures:
        prediction_curves[temperature] = predictions

sensitivity_df = pd.DataFrame(results)
sensitivity_path = RESULTS_DIR / "temperature_sensitivity_metrics.csv"
sensitivity_df.to_csv(sensitivity_path, index=False)

summary_df = sensitivity_df[
    sensitivity_df["Temperature_C"].isin(scenario_temperatures)
].copy()

summary_path = RESULTS_DIR / "temperature_sensitivity_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nTemperature sensitivity summary:")
print(
    summary_df[
        ["Temperature_C", "Model", "Mean_Predicted_Demand"]
    ].to_string(index=False)
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.ravel()

for ax, model_name in zip(
    axes,
    ["Linear Regression", "Random Forest", "XGBoost", "LSTM"],
):
    model_results = sensitivity_df[
        sensitivity_df["Model"] == model_name
    ]

    ax.plot(
        model_results["Temperature_C"],
        model_results["Mean_Predicted_Demand"],
        marker="o",
        markersize=3,
    )
    ax.set_title(model_name)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Mean predicted demand")
    ax.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = RESULTS_DIR / "temperature_sensitivity_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSaved results to: {RESULTS_DIR}")
