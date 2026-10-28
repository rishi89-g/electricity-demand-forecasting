
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

DEMAND_FILE = DATA_DIR / "hourly_electricity_demand.csv"
TEMPERATURE_FILE = DATA_DIR / "hourly_temprature_data.csv"


def load_demand_data() -> pd.DataFrame:
    if not DEMAND_FILE.is_file():
        raise FileNotFoundError(f"Demand dataset not found: {DEMAND_FILE}")

    data = pd.read_csv(DEMAND_FILE, sep=";")

    required_columns = {"startTime", "endTime"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing demand timestamp columns: {sorted(missing_columns)}"
        )

    data["startTime"] = pd.to_datetime(data["startTime"], errors="coerce", utc=True)
    data["endTime"] = pd.to_datetime(data["endTime"], errors="coerce", utc=True)

    return data


def load_temperature_data() -> pd.DataFrame:
    if not TEMPERATURE_FILE.is_file():
        raise FileNotFoundError(
            f"Temperature dataset not found: {TEMPERATURE_FILE}"
        )

    data = pd.read_csv(TEMPERATURE_FILE)

    required_columns = {"Year", "Month", "Day", "Time [Local time]"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing temperature timestamp columns: {sorted(missing_columns)}"
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

    return data


def report_timestamps(name: str, timestamps: pd.Series) -> None:
    valid = timestamps.dropna()

    print(f"\n{'=' * 65}")
    print(f"{name.upper()} TIMESTAMP CHECK")
    print(f"{'=' * 65}")
    print(f"Total records: {len(timestamps)}")
    print(f"Invalid timestamps: {timestamps.isna().sum()}")
    print(f"Duplicate timestamps: {valid.duplicated().sum()}")

    if not valid.empty:
        print(f"First timestamp: {valid.min()}")
        print(f"Last timestamp: {valid.max()}")
        print(f"Unique timestamps: {valid.nunique()}")


def main() -> None:
    demand = load_demand_data()
    temperature = load_temperature_data()

    report_timestamps("Electricity demand (startTime)", demand["startTime"])
    report_timestamps("Electricity demand (endTime)", demand["endTime"])
    report_timestamps("Temperature (local time)", temperature["timestamp_local"])

    print(f"\n{'=' * 65}")
    print("TIMESTAMP VALIDATION COMPLETED")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
