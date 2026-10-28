
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
TEMPERATURE_FILE = DATA_DIR / "hourly_temprature_data_2.csv"


def main() -> None:
    if not TEMPERATURE_FILE.is_file():
        raise FileNotFoundError(
            f"Temperature dataset not found: {TEMPERATURE_FILE}"
        )

    data = pd.read_csv(TEMPERATURE_FILE)

    timestamp_text = (
        data["Year"].astype("Int64").astype(str)
        + "-"
        + data["Month"].astype("Int64").astype(str).str.zfill(2)
        + "-"
        + data["Day"].astype("Int64").astype(str).str.zfill(2)
        + " "
        + data["Time [Local time]"].astype(str).str.strip()
    )

    data["timestamp_local"] = pd.to_datetime(
        timestamp_text,
        errors="coerce",
    )

    valid_data = data.dropna(subset=["timestamp_local"]).copy()

    duplicate_rows = valid_data[
        valid_data["timestamp_local"].duplicated(keep=False)
    ].sort_values("timestamp_local")

    expected_index = pd.date_range(
        start="2022-01-01 00:00:00",
        end="2022-12-31 23:00:00",
        freq="h",
    )

    observed_index = pd.DatetimeIndex(
        valid_data["timestamp_local"].drop_duplicates().sort_values()
    )

    missing_timestamps = expected_index.difference(observed_index)

    print("=" * 65)
    print("TEMPERATURE DATA QUALITY DETAILS")
    print("=" * 65)

    print("\nDuplicate timestamp records:")
    if duplicate_rows.empty:
        print("None")
    else:
        columns_to_show = [
            "timestamp_local",
            "Observation station",
            "Average temperature [°C]",
        ]
        print(duplicate_rows[columns_to_show].to_string(index=False))

    print("\nMissing timestamps:")
    if len(missing_timestamps) == 0:
        print("None")
    else:
        for timestamp in missing_timestamps:
            print(timestamp)

    print("\nObservations surrounding missing timestamps:")
    for timestamp in missing_timestamps:
        start = timestamp - pd.Timedelta(hours=2)
        end = timestamp + pd.Timedelta(hours=2)

        surrounding = valid_data.loc[
            valid_data["timestamp_local"].between(start, end),
            [
                "timestamp_local",
                "Observation station",
                "Average temperature [°C]",
            ],
        ].sort_values("timestamp_local")

        print(f"\nMissing hour: {timestamp}")
        print(surrounding.to_string(index=False))

    print("\nTEMPERATURE DATA QUALITY INSPECTION COMPLETED")


if __name__ == "__main__":
    main()
