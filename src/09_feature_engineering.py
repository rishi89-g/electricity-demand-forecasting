
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "merged_demand_temperature.csv"
OUTPUT_FILE = PROCESSED_DIR / "feature_engineered_data.csv"

TARGET = "Electricity consumption in Finland"
TEMPERATURE = "Average temperature [°C]"


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Merged dataset not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE, parse_dates=["timestamp_utc"])

    required_columns = {"timestamp_utc", TARGET, TEMPERATURE}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["timestamp_utc"] = pd.to_datetime(
        data["timestamp_utc"], utc=True, errors="coerce"
    )

    if data["timestamp_utc"].isna().any():
        raise ValueError("Invalid timestamps found in the merged dataset.")

    data = data.sort_values("timestamp_utc").reset_index(drop=True)

    if data["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps found before feature engineering.")

    data["hour"] = data["timestamp_utc"].dt.hour
    data["day_of_week"] = data["timestamp_utc"].dt.dayofweek
    data["month"] = data["timestamp_utc"].dt.month
    data["day_of_month"] = data["timestamp_utc"].dt.day
    data["is_weekend"] = (
        data["timestamp_utc"].dt.dayofweek >= 5
    ).astype(int)

    data["hour_sin"] = (
        __import__("numpy").sin(2 * __import__("numpy").pi * data["hour"] / 24)
    )
    data["hour_cos"] = (
        __import__("numpy").cos(2 * __import__("numpy").pi * data["hour"] / 24)
    )

    data["day_of_week_sin"] = (
        __import__("numpy").sin(
            2 * __import__("numpy").pi * data["day_of_week"] / 7
        )
    )
    data["day_of_week_cos"] = (
        __import__("numpy").cos(
            2 * __import__("numpy").pi * data["day_of_week"] / 7
        )
    )

    for lag in (1, 2, 24, 168):
        data[f"lag_{lag}"] = data[TARGET].shift(lag)

    data["rolling_mean_24"] = data[TARGET].shift(1).rolling(24).mean()
    data["rolling_mean_168"] = data[TARGET].shift(1).rolling(168).mean()

    initial_rows = len(data)
    data = data.dropna().reset_index(drop=True)
    removed_rows = initial_rows - len(data)

    if data.isna().any().any():
        raise ValueError("Missing values remain after feature engineering.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)

    print("=" * 65)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 65)
    print(f"Rows before feature engineering: {initial_rows}")
    print(f"Rows after removing lag-related missing values: {len(data)}")
    print(f"Rows removed: {removed_rows}")
    print(f"Final dataset shape: {data.shape}")

    print("\nFeature columns:")
    print(data.columns.tolist())

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    print(f"\nSaved dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
