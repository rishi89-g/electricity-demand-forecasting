
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "merged_demand_temperature.csv"
)

DEMAND_COLUMN = "Electricity consumption in Finland"
TEMPERATURE_COLUMN = "Average temperature [°C]"


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Merged dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE, parse_dates=["timestamp_utc"])

    required_columns = {
        "timestamp_utc",
        "timestamp_local",
        DEMAND_COLUMN,
        TEMPERATURE_COLUMN,
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["timestamp_utc"] = pd.to_datetime(
        data["timestamp_utc"], utc=True, errors="coerce"
    )

    print("=" * 65)
    print("MERGED DATASET VALIDATION")
    print("=" * 65)
    print(f"File: {DATA_FILE.name}")
    print(f"Dataset shape: {data.shape}")
    print(f"Invalid UTC timestamps: {data['timestamp_utc'].isna().sum()}")
    print(f"Duplicate UTC timestamps: {data['timestamp_utc'].duplicated().sum()}")

    valid_timestamps = data["timestamp_utc"].dropna().sort_values()
    time_differences = valid_timestamps.diff().dropna()
    gap_count = int((time_differences != pd.Timedelta(hours=1)).sum())

    print(f"First UTC timestamp: {valid_timestamps.min()}")
    print(f"Last UTC timestamp: {valid_timestamps.max()}")
    print(f"Non-hourly intervals: {gap_count}")

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    print("\nDuplicate complete rows:", data.duplicated().sum())

    print("\nDescriptive statistics:")
    print(
        data[[DEMAND_COLUMN, TEMPERATURE_COLUMN]]
        .describe()
        .round(4)
        .to_string()
    )

    print("\nFirst and last three records:")
    print(
        pd.concat([data.head(3), data.tail(3)])
        [
            [
                "timestamp_utc",
                "timestamp_local",
                DEMAND_COLUMN,
                TEMPERATURE_COLUMN,
            ]
        ]
        .to_string(index=False)
    )

    print("\nMERGED DATASET VALIDATION COMPLETED")


if __name__ == "__main__":
    main()
