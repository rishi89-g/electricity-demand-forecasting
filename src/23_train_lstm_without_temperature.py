
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

target = "Electricity consumption in Finland"
sequence_length = 24

features = [
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
validation_data = pd.read_csv(PROCESSED_DIR / "validation_data.csv")
test_data = pd.read_csv(PROCESSED_DIR / "test_data.csv")

X_train = train_data[features].to_numpy()
y_train = train_data[[target]].to_numpy()

X_validation = validation_data[features].to_numpy()
y_validation = validation_data[[target]].to_numpy()

X_test = test_data[features].to_numpy()
y_test = test_data[[target]].to_numpy()

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(X_train)
X_validation_scaled = feature_scaler.transform(X_validation)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train)
y_validation_scaled = target_scaler.transform(y_validation)
y_test_scaled = target_scaler.transform(y_test)


def create_sequences(X, y, length):
    X_sequences = []
    y_sequences = []

    for i in range(length, len(X) + 1):
        X_sequences.append(X[i - length:i])
        y_sequences.append(y[i - 1])

    return np.asarray(X_sequences), np.asarray(y_sequences)


X_train_seq, y_train_seq = create_sequences(
    X_train_scaled, y_train_scaled, sequence_length
)
X_validation_seq, y_validation_seq = create_sequences(
    X_validation_scaled, y_validation_scaled, sequence_length
)
X_test_seq, y_test_seq = create_sequences(
    X_test_scaled, y_test_scaled, sequence_length
)

print("Training LSTM model...")

model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(sequence_length, len(features)),
    ),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

model.fit(
    X_train_seq,
    y_train_seq,
    validation_data=(X_validation_seq, y_validation_seq),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)

predictions_scaled = model.predict(X_test_seq, verbose=0)
predictions = target_scaler.inverse_transform(predictions_scaled)
actual = target_scaler.inverse_transform(y_test_seq)

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

model.save(RESULTS_DIR / "lstm_model_without_temperature.keras")
joblib.dump(
    feature_scaler,
    RESULTS_DIR / "lstm_feature_scaler_without_temperature.joblib",
)
joblib.dump(
    target_scaler,
    RESULTS_DIR / "lstm_target_scaler_without_temperature.joblib",
)

pd.DataFrame({
    "Actual": actual.ravel(),
    "Predicted": predictions.ravel(),
}).to_csv(
    RESULTS_DIR / "lstm_without_temperature_predictions.csv",
    index=False,
)

pd.DataFrame([{
    "Model": "LSTM without temperature",
    "Test_MAE": mae,
    "Test_RMSE": rmse,
}]).to_csv(
    RESULTS_DIR / "lstm_without_temperature_results.csv",
    index=False,
)

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Results saved to: {RESULTS_DIR}")
