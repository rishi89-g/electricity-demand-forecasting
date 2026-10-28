
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "merged_demand_temperature.csv"
)


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Merged dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    data["timestamp_utc"] = pd.to_datetime(
        data["timestamp_utc"], utc=True, errors="coerce"
    )

    timestamps = data["timestamp_utc"].dropna().sort_values()

    print("=" * 65)
    print("HOURLY TIMESTAMP CONTINUITY CHECK")
    print("=" * 65)
    print(f"Total records: {len(data)}")
    print(f"Invalid timestamps: {data['timestamp_utc'].isna().sum()}")
    print(f"Duplicate timestamps: {timestamps.duplicated().sum()}")

    differences = timestamps.diff()
    irregular = differences.notna() & (differences != pd.Timedelta(hours=1))

    print(f"Non-hourly intervals: {int(irregular.sum())}")

    if irregular.any():
        print("\nIrregular intervals:")
        positions = differences.index[irregular]

        for position in positions[:20]:
            current_position = data.index[data["timestamp_utc"] == timestamps.loc[position]]
            print(
                f"Timestamp: {timestamps.loc[position]}, "
                f"interval: {differences.loc[position]}"
            )

    if not timestamps.empty:
        expected = pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq="h",
        )
        missing_hours = expected.difference(pd.DatetimeIndex(timestamps))

        print(f"\nExpected hourly timestamps in observed range: {len(expected)}")
        print(f"Missing UTC timestamps: {len(missing_hours)}")

        if len(missing_hours):
            print("\nFirst 20 missing timestamps:")
            print(missing_hours[:20].to_list())

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    print("\nHOURLY CONTINUITY CHECK COMPLETED")


if __name__ == "__main__":
    main()
