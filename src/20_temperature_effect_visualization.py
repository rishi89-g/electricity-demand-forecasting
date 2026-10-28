
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

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

print("Loading data...")

train_data = pd.read_csv(PROCESSED_DIR / "train_data.csv")
test_data = pd.read_csv(PROCESSED_DIR / "test_data.csv")

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

scenario_temperatures = [-15, 0, 10]
scenario_data = {}

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

for temperature in scenario_temperatures:
    X_scenario = X_test.copy()
    X_scenario[temperature_col] = temperature

    scenario_data[temperature] = {
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

    lstm_predictions = lstm_model.predict(
        sequences,
        verbose=0,
    )

    scenario_data[temperature]["LSTM"] = (
        target_scaler.inverse_transform(
            lstm_predictions.reshape(-1, 1)
        ).ravel()
    )

print("Generating temperature scenario plots...")

models = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]
fig, axes = plt.subplots(
    len(models), 1, figsize=(14, 16), sharex=True
)

plot_count = min(168, len(X_test) - 23)
x_axis = np.arange(plot_count)
actual_lstm = y_test.iloc[23:23 + plot_count].to_numpy()

for ax, model_name in zip(axes, models):
    if model_name == "LSTM":
        actual = actual_lstm
    else:
        actual = y_test.iloc[:plot_count].to_numpy()

    ax.plot(
        x_axis,
        actual[:plot_count],
        label="Actual demand",
        linewidth=1.5,
    )

    for temperature in scenario_temperatures:
        predictions = scenario_data[temperature][model_name]
        offset = 23 if model_name == "LSTM" else 0

        ax.plot(
            x_axis,
            predictions[:plot_count],
            label=f"{temperature}°C",
            alpha=0.85,
        )

    ax.set_title(model_name)
    ax.set_ylabel("Electricity demand")
    ax.legend(ncol=4)

axes[-1].set_xlabel("Test observations")
plt.tight_layout()

output_path = RESULTS_DIR / "temperature_effect_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot: {output_path}")
