


import os
import random
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. File paths
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
# 3. Configuration
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

SEQUENCE_LENGTH = 24
EPOCHS = 50
BATCH_SIZE = 32


# ============================================================
# 4. Load datasets
# ============================================================

train_data = pd.read_csv(TRAIN_PATH)
validation_data = pd.read_csv(VALIDATION_PATH)
test_data = pd.read_csv(TEST_PATH)

print("Datasets loaded successfully.")
print("Training data shape:", train_data.shape)
print("Validation data shape:", validation_data.shape)
print("Test data shape:", test_data.shape)


# ============================================================
# 5. Prepare arrays
# ============================================================

X_train_raw = train_data[FEATURE_COLUMNS].values
y_train_raw = train_data[TARGET].values.reshape(-1, 1)

X_validation_raw = validation_data[FEATURE_COLUMNS].values
y_validation_raw = validation_data[TARGET].values.reshape(-1, 1)

X_test_raw = test_data[FEATURE_COLUMNS].values
y_test_raw = test_data[TARGET].values.reshape(-1, 1)


# ============================================================
# 6. Scale features and target
# Fit scalers ONLY on training data
# ============================================================

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

X_train_scaled = feature_scaler.fit_transform(X_train_raw)

X_validation_scaled = feature_scaler.transform(
    X_validation_raw
)

X_test_scaled = feature_scaler.transform(
    X_test_raw
)

y_train_scaled = target_scaler.fit_transform(
    y_train_raw
).flatten()

y_validation_scaled = target_scaler.transform(
    y_validation_raw
).flatten()

y_test_scaled = target_scaler.transform(
    y_test_raw
).flatten()


# ============================================================
# 7. Create time-series sequences
# ============================================================

def create_sequences(X, y, sequence_length):
    X_sequences = []
    y_values = []

    for i in range(sequence_length, len(X)):
        X_sequences.append(
            X[i - sequence_length:i]
        )
        y_values.append(y[i])

    return np.array(X_sequences), np.array(y_values)


# Training sequences
X_train, y_train = create_sequences(
    X_train_scaled,
    y_train_scaled,
    SEQUENCE_LENGTH
)


# Validation sequences:
# Include the last 24 training observations as context.

X_validation_with_context = np.concatenate(
    [
        X_train_scaled[-SEQUENCE_LENGTH:],
        X_validation_scaled
    ],
    axis=0
)

y_validation_with_context = np.concatenate(
    [
        y_train_scaled[-SEQUENCE_LENGTH:],
        y_validation_scaled
    ],
    axis=0
)

X_validation, y_validation = create_sequences(
    X_validation_with_context,
    y_validation_with_context,
    SEQUENCE_LENGTH
)


# Test sequences:
# Include the last 24 validation observations as context.

X_test_with_context = np.concatenate(
    [
        X_validation_scaled[-SEQUENCE_LENGTH:],
        X_test_scaled
    ],
    axis=0
)

y_test_with_context = np.concatenate(
    [
        y_validation_scaled[-SEQUENCE_LENGTH:],
        y_test_scaled
    ],
    axis=0
)

X_test, y_test = create_sequences(
    X_test_with_context,
    y_test_with_context,
    SEQUENCE_LENGTH
)


print("\nSequence shapes:")
print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)
print("X_test:", X_test.shape)


# ============================================================
# 8. Build LSTM model
# ============================================================

model = Sequential([
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(
            SEQUENCE_LENGTH,
            len(FEATURE_COLUMNS)
        )
    ),

    Dropout(0.2),

    LSTM(
        units=32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()


# ============================================================
# 9. Early stopping
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)


# ============================================================
# 10. Train LSTM
# ============================================================

print("\nTraining LSTM model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_validation,
        y_validation
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

print("Training completed.")


# ============================================================
# 10A. SAVE MODEL AND SCALERS
# Required for temperature response analysis
# ============================================================

model_path = os.path.join(
    RESULTS_DIR,
    "lstm_model_with_temperature.keras"
)

feature_scaler_path = os.path.join(
    RESULTS_DIR,
    "lstm_feature_scaler_with_temperature.joblib"
)

target_scaler_path = os.path.join(
    RESULTS_DIR,
    "lstm_target_scaler_with_temperature.joblib"
)

# Save the trained LSTM model
model.save(model_path)

# Save the feature scaler fitted on training data
joblib.dump(
    feature_scaler,
    feature_scaler_path
)

# Save the target scaler fitted on training data
joblib.dump(
    target_scaler,
    target_scaler_path
)

print("\nModel and scalers saved successfully:")
print(model_path)
print(feature_scaler_path)
print(target_scaler_path)


# ============================================================
# 11. Generate predictions
# ============================================================

validation_predictions_scaled = model.predict(
    X_validation,
    verbose=0
).flatten()

test_predictions_scaled = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# 12. Inverse transform predictions
# ============================================================

validation_predictions = target_scaler.inverse_transform(
    validation_predictions_scaled.reshape(-1, 1)
).flatten()

test_predictions = target_scaler.inverse_transform(
    test_predictions_scaled.reshape(-1, 1)
).flatten()

y_validation_actual = target_scaler.inverse_transform(
    y_validation.reshape(-1, 1)
).flatten()

y_test_actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()


# ============================================================
# 13. Evaluation function
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
# 14. Evaluate model
# ============================================================

validation_mae, validation_rmse = evaluate_model(
    y_validation_actual,
    validation_predictions,
    "Validation"
)

test_mae, test_rmse = evaluate_model(
    y_test_actual,
    test_predictions,
    "Test"
)


# ============================================================
# 15. Save metrics
# ============================================================

results = pd.DataFrame({
    "Model": ["LSTM"],
    "Validation_MAE": [validation_mae],
    "Validation_RMSE": [validation_rmse],
    "Test_MAE": [test_mae],
    "Test_RMSE": [test_rmse]
})

results_path = os.path.join(
    RESULTS_DIR,
    "lstm_results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print(f"\nResults saved to: {results_path}")


# ============================================================
# 16. Save predictions
# ============================================================

test_timestamps = test_data[
    "timestamp_utc"
].reset_index(drop=True)

prediction_results = pd.DataFrame({
    "timestamp_utc": test_timestamps,
    "actual_demand": y_test_actual,
    "predicted_demand": test_predictions
})

prediction_path = os.path.join(
    RESULTS_DIR,
    "lstm_predictions.csv"
)

prediction_results.to_csv(
    prediction_path,
    index=False
)

print(f"Predictions saved to: {prediction_path}")


# ============================================================
# 17. Plot actual vs predicted demand
# ============================================================

plot_points = min(168, len(y_test_actual))

plt.figure(figsize=(14, 6))

plt.plot(
    y_test_actual[:plot_points],
    label="Actual Demand"
)

plt.plot(
    test_predictions[:plot_points],
    label="Predicted Demand"
)

plt.title(
    "LSTM: Actual vs Predicted Electricity Demand"
)

plt.xlabel("Test Time Steps")
plt.ylabel("Electricity Demand")

plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(
    RESULTS_DIR,
    "lstm_forecast_test.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.show()

print(f"Plot saved to: {plot_path}")


# ============================================================
# 18. Plot training history
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

history_path = os.path.join(
    RESULTS_DIR,
    "lstm_training_history.png"
)

plt.savefig(
    history_path,
    dpi=300
)

plt.show()

print(f"Training history saved to: {history_path}")

print("\nLSTM execution completed successfully.")
