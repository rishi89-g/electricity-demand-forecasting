
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_FILE = PROCESSED_DIR / "train_data.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation_data.csv"
TEST_FILE = PROCESSED_DIR / "test_data.csv"

TARGET = "Electricity consumption in Finland"
NAIVE_FEATURE = "lag_24"


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    }


def main() -> None:
    for file_path in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE):
        if not file_path.is_file():
            raise FileNotFoundError(f"Required dataset not found: {file_path}")

    validation_data = pd.read_csv(VALIDATION_FILE)
    test_data = pd.read_csv(TEST_FILE)

    for name, data in (
        ("Validation", validation_data),
        ("Test", test_data),
    ):
        missing_columns = {TARGET, NAIVE_FEATURE}.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"{name} dataset is missing columns: {sorted(missing_columns)}"
            )

    validation_actual = validation_data[TARGET]
    validation_predictions = validation_data[NAIVE_FEATURE]

    test_actual = test_data[TARGET]
    test_predictions = test_data[NAIVE_FEATURE]

    validation_metrics = calculate_metrics(
        validation_actual,
        validation_predictions,
    )
    test_metrics = calculate_metrics(test_actual, test_predictions)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {"Dataset": "Validation", **validation_metrics},
            {"Dataset": "Test", **test_metrics},
        ]
    )

    results_file = RESULTS_DIR / "naive_results.csv"
    predictions_file = RESULTS_DIR / "naive_predictions.csv"
    plot_file = RESULTS_DIR / "naive_forecast_test.png"

    results.to_csv(results_file, index=False)

    predictions = pd.DataFrame(
        {
            "Actual": test_actual,
            "Predicted": test_predictions,
        }
    )
    predictions.to_csv(predictions_file, index=False)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.plot(test_actual.to_numpy(), label="Actual", linewidth=1)
    plt.plot(test_predictions.to_numpy(), label="Naive forecast", linewidth=1)
    plt.title("Naive Forecast: Electricity Demand on Test Set")
    plt.xlabel("Test observation")
    plt.ylabel("Electricity consumption in Finland")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print("=" * 65)
    print("NAIVE FORECAST EVALUATION")
    print("=" * 65)
    print("\nValidation metrics:")
    print(pd.Series(validation_metrics).round(4).to_string())
    print("\nTest metrics:")
    print(pd.Series(test_metrics).round(4).to_string())
    print(f"\nResults saved: {results_file}")
    print(f"Predictions saved: {predictions_file}")
    print(f"Forecast plot saved: {plot_file}")


if __name__ == "__main__":
    main()
