
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "feature_engineered_data.csv"
TRAIN_FILE = PROCESSED_DIR / "train_data.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation_data.csv"
TEST_FILE = PROCESSED_DIR / "test_data.csv"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Feature-engineered dataset not found: {INPUT_FILE}")

    data = pd.read_csv(INPUT_FILE, parse_dates=["timestamp_utc"])

    data["timestamp_utc"] = pd.to_datetime(
        data["timestamp_utc"],
        utc=True,
        errors="coerce",
    )

    if data["timestamp_utc"].isna().any():
        raise ValueError("Invalid timestamps found in the input dataset.")

    data = data.sort_values("timestamp_utc").reset_index(drop=True)

    if data["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps found in the input dataset.")

    total_rows = len(data)
    train_end = int(total_rows * TRAIN_RATIO)
    validation_end = int(total_rows * (TRAIN_RATIO + VALIDATION_RATIO))

    train_data = data.iloc[:train_end].copy()
    validation_data = data.iloc[train_end:validation_end].copy()
    test_data = data.iloc[validation_end:].copy()

    if not (
        train_data["timestamp_utc"].max()
        < validation_data["timestamp_utc"].min()
        and validation_data["timestamp_utc"].max()
        < test_data["timestamp_utc"].min()
    ):
        raise ValueError("Chronological split validation failed.")

    train_data.to_csv(TRAIN_FILE, index=False)
    validation_data.to_csv(VALIDATION_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    print("=" * 65)
    print("CHRONOLOGICAL DATA SPLIT COMPLETED")
    print("=" * 65)
    print(f"Total rows: {total_rows}")
    print(f"Training rows: {len(train_data)} ({len(train_data) / total_rows:.2%})")
    print(
        f"Validation rows: {len(validation_data)} "
        f"({len(validation_data) / total_rows:.2%})"
    )
    print(f"Testing rows: {len(test_data)} ({len(test_data) / total_rows:.2%})")

    print("\nData periods (UTC):")
    print(
        f"Training: {train_data['timestamp_utc'].min()} "
        f"to {train_data['timestamp_utc'].max()}"
    )
    print(
        f"Validation: {validation_data['timestamp_utc'].min()} "
        f"to {validation_data['timestamp_utc'].max()}"
    )
    print(
        f"Testing: {test_data['timestamp_utc'].min()} "
        f"to {test_data['timestamp_utc'].max()}"
    )

    print("\nChronological order verified successfully.")
    print(f"Training data saved: {TRAIN_FILE}")
    print(f"Validation data saved: {VALIDATION_FILE}")
    print(f"Testing data saved: {TEST_FILE}")


if __name__ == "__main__":
    main()
