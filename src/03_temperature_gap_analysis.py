
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

TEMPERATURE_FILE = DATA_DIR / "hourly_temprature_data.csv"


def main() -> None:
    if not TEMPERATURE_FILE.is_file():
        raise FileNotFoundError(
            f"Temperature dataset not found: {TEMPERATURE_FILE}"
        )

    data = pd.read_csv(TEMPERATURE_FILE)

    required_columns = {
        "Year",
        "Month",
        "Day",
        "Time [Local time]",
        "Average temperature [°C]",
    }
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    date_string = (
        data["Year"].astype("Int64").astype(str)
        + "-"
        + data["Month"].astype("Int64").astype(str).str.zfill(2)
        + "-"
        + data["Day"].astype("Int64").astype(str).str.zfill(2)
        + " "
        + data["Time [Local time]"].astype(str).str.strip()
    )

    data["timestamp_local"] = pd.to_datetime(date_string, errors="coerce")

    invalid_count = data["timestamp_local"].isna().sum()
    valid_data = data.dropna(subset=["timestamp_local"]).copy()

    duplicate_mask = valid_data["timestamp_local"].duplicated(keep=False)
    duplicates = valid_data.loc[
        duplicate_mask,
        ["timestamp_local", "Observation station", "Average temperature [°C]"],
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
    unexpected_timestamps = observed_index.difference(expected_index)

    print("=" * 65)
    print("TEMPERATURE TIMESTAMP GAP ANALYSIS")
    print("=" * 65)
    print(f"Total source records: {len(data)}")
    print(f"Invalid timestamps: {invalid_count}")
    print(f"Duplicate timestamp rows: {duplicate_mask.sum()}")
    print(f"Unique valid timestamps: {valid_data['timestamp_local'].nunique()}")
    print(f"Expected hourly timestamps in 2022: {len(expected_index)}")
    print(f"Missing hourly timestamps: {len(missing_timestamps)}")
    print(f"Unexpected timestamps outside 2022: {len(unexpected_timestamps)}")

    print("\nDuplicate timestamp details:")
    if duplicates.empty:
        print("None")
    else:
        print(duplicates.to_string(index=False))

    print("\nFirst 20 missing timestamps:")
    if len(missing_timestamps) == 0:
        print("None")
    else:
        print(missing_timestamps[:20].to_list())

    print("\nFirst 20 unexpected timestamps:")
    if len(unexpected_timestamps) == 0:
        print("None")
    else:
        print(unexpected_timestamps[:20].to_list())

    print("\nMissing timestamps by month:")
    if len(missing_timestamps) > 0:
        missing_by_month = pd.Series(
            missing_timestamps.month
        ).value_counts().sort_index()
        print(missing_by_month.to_string())
    else:
        print("No missing timestamps.")

    print("\nTemperature missing values:")
    print(
        valid_data["Average temperature [°C]"]
        .isna()
        .sum()
    )

    print("\nTEMPERATURE GAP ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
