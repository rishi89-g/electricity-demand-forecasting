
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

target = "Electricity consumption in Finland"
temperature_col = "Average temperature [°C]"
sequence_length = 24
temperatures = [-15, 0, 10, 20, 30]

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

print("Loading data and saved LSTM model...")

test_data = pd.read_csv(PROCESSED_DIR / "test_data.csv")
model = load_model(RESULTS_DIR / "lstm_model_with_temperature.keras")
feature_scaler = joblib.load(
    RESULTS_DIR / "lstm_feature_scaler_with_temperature.joblib"
)
target_scaler = joblib.load(
    RESULTS_DIR / "lstm_target_scaler_with_temperature.joblib"
)

X_test = test_data[features].copy()
y_test = test_data[target].to_numpy()

if len(X_test) < sequence_length:
    raise ValueError("Test data must contain at least 24 observations.")

results = []
prediction_data = {}

print("Evaluating temperature scenarios...")

for temperature in temperatures:
    X_scenario = X_test.copy()
    X_scenario[temperature_col] = temperature

    X_scaled = feature_scaler.transform(X_scenario.to_numpy())

    sequences = np.stack([
        X_scaled[i - sequence_length + 1:i + 1]
        for i in range(sequence_length - 1, len(X_scaled))
    ])

    predictions_scaled = model.predict(sequences, verbose=0)
    predictions = target_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).ravel()

    prediction_data[temperature] = predictions

    results.append({
        "Temperature_C": temperature,
        "Mean_Predicted_Demand": float(np.mean(predictions)),
        "Min_Predicted_Demand": float(np.min(predictions)),
        "Max_Predicted_Demand": float(np.max(predictions)),
        "MAE": float(
            np.mean(np.abs(y_test[sequence_length - 1:] - predictions))
        ),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    (y_test[sequence_length - 1:] - predictions) ** 2
                )
            )
        ),
    })

summary = pd.DataFrame(results)
summary.to_csv(
    RESULTS_DIR / "lstm_temperature_response_summary.csv",
    index=False,
)

prediction_df = pd.DataFrame({
    "Actual": y_test[sequence_length - 1:],
})

for temperature, predictions in prediction_data.items():
    prediction_df[f"Predicted_{temperature}C"] = predictions

prediction_df.to_csv(
    RESULTS_DIR / "lstm_temperature_response_predictions.csv",
    index=False,
)

plt.figure(figsize=(12, 6))

for temperature, predictions in prediction_data.items():
    plt.plot(
        predictions[:168],
        label=f"{temperature}°C",
    )

plt.plot(
    y_test[sequence_length - 1:sequence_length - 1 + 168],
    label="Actual demand",
    linewidth=1.5,
)

plt.xlabel("Test observations")
plt.ylabel("Electricity demand")
plt.title("LSTM Temperature Response")
plt.legend()
plt.tight_layout()

plot_path = RESULTS_DIR / "lstm_temperature_response.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nLSTM temperature response summary:")
print(summary.to_string(index=False))
print(f"\nResults saved to: {RESULTS_DIR}")
