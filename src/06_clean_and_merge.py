
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEMAND_FILE = RAW_DIR / "hourly_electricity_demand.csv"
TEMPERATURE_FILE = RAW_DIR / "hourly_temprature_data_2.csv"
OUTPUT_FILE = PROCESSED_DIR / "merged_demand_temperature.csv"

TIMEZONE = "Europe/Helsinki"
TEMPERATURE_COLUMN = "Average temperature [°C]"


def load_demand_data() -> pd.DataFrame:
    if not DEMAND_FILE.is_file():
        raise FileNotFoundError(f"Demand file not found: {DEMAND_FILE}")

    demand = pd.read_csv(DEMAND_FILE, sep=";")

    required = {"startTime", "Electricity consumption in Finland"}
    missing = required.difference(demand.columns)
    if missing:
        raise ValueError(f"Missing demand columns: {sorted(missing)}")

    demand["timestamp_utc"] = pd.to_datetime(
        demand["startTime"], errors="coerce", utc=True
    )

    if demand["timestamp_utc"].isna().any():
        raise ValueError("Demand data contains invalid timestamps.")

    if demand["timestamp_utc"].duplicated().any():
        raise ValueError("Demand data contains duplicate UTC timestamps.")

    demand["timestamp_local"] = demand["timestamp_utc"].dt.tz_convert(TIMEZONE)

    return demand


def load_temperature_data() -> pd.DataFrame:
    if not TEMPERATURE_FILE.is_file():
        raise FileNotFoundError(
            f"Temperature file not found: {TEMPERATURE_FILE}"
        )

    temperature = pd.read_csv(TEMPERATURE_FILE)

    required = {
        "Year", "Month", "Day", "Time [Local time]", TEMPERATURE_COLUMN
    }
    missing = required.difference(temperature.columns)
    if missing:
        raise ValueError(f"Missing temperature columns: {sorted(missing)}")

    timestamp_text = (
        temperature["Year"].astype("Int64").astype(str)
        + "-"
        + temperature["Month"].astype("Int64").astype(str).str.zfill(2)
        + "-"
        + temperature["Day"].astype("Int64").astype(str).str.zfill(2)
        + " "
        + temperature["Time [Local time]"].astype(str).str.strip()
    )

    temperature["timestamp_local_naive"] = pd.to_datetime(
        timestamp_text, errors="coerce"
    )

    if temperature["timestamp_local_naive"].isna().any():
        raise ValueError("Temperature data contains invalid local timestamps.")

    duplicate_mask = temperature["timestamp_local_naive"].duplicated(
        keep=False
    )
    duplicate_rows = temperature.loc[duplicate_mask].copy()

    print("=" * 65)
    print("TEMPERATURE TIMESTAMP REVIEW")
    print("=" * 65)
    print(f"Source records: {len(temperature)}")
    print(f"Rows with duplicated local timestamps: {duplicate_mask.sum()}")

    if not duplicate_rows.empty:
        print("\nDuplicated local timestamp records:")
        print(
            duplicate_rows[
                [
                    "timestamp_local_naive",
                    TEMPERATURE_COLUMN,
                ]
            ].sort_values("timestamp_local_naive").to_string(index=False)
        )

    # The spring transition creates a nonexistent local hour.
    # Keep the source records and explicitly report localization outcomes.
    localized = temperature["timestamp_local_naive"].dt.tz_localize(
        TIMEZONE,
        ambiguous="NaT",
        nonexistent="NaT",
    )

    temperature["timestamp_local"] = localized
    invalid_localization = temperature["timestamp_local"].isna()

    print(
        "\nRows that cannot be uniquely localized:",
        int(invalid_localization.sum()),
    )

    if invalid_localization.any():
        print(
            temperature.loc[
                invalid_localization,
                ["timestamp_local_naive", TEMPERATURE_COLUMN],
            ].to_string(index=False)
        )

    # A repeated autumn local hour cannot be assigned to one of the two
    # UTC instants reliably without additional source information.
    # Exclude unresolved rows rather than guessing their UTC timestamps.
    temperature = temperature.dropna(subset=["timestamp_local"]).copy()

    temperature["timestamp_utc"] = (
        temperature["timestamp_local"].dt.tz_convert("UTC")
    )

    # If duplicate UTC timestamps remain, combine their temperature
    # observations into one value.
    temperature = (
        temperature.groupby("timestamp_utc", as_index=False)
        .agg({TEMPERATURE_COLUMN: "mean"})
        .sort_values("timestamp_utc")
    )

    if temperature["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate UTC timestamps remain in temperature data.")

    return temperature


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    demand = load_demand_data()
    temperature = load_temperature_data()

    merged = demand.merge(
        temperature,
        on="timestamp_utc",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_before = int((merged["_merge"] == "left_only").sum())
    merged = merged.drop(columns="_merge")
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)

    missing_before = int(merged[TEMPERATURE_COLUMN].isna().sum())

    print("\n" + "=" * 65)
    print("MERGE VALIDATION")
    print("=" * 65)
    print(f"Demand records: {len(demand)}")
    print(f"Unique temperature UTC timestamps: {len(temperature)}")
    print(f"Merged records: {len(merged)}")
    print(f"Demand timestamps without temperature matches: {unmatched_before}")
    print(f"Missing temperature values before interpolation: {missing_before}")

    # Interpolate only short, internal gaps in chronological order.
    merged[TEMPERATURE_COLUMN] = merged[TEMPERATURE_COLUMN].interpolate(
        method="linear",
        limit=2,
        limit_area="inside",
    )

    missing_after = int(merged[TEMPERATURE_COLUMN].isna().sum())

    print(f"Missing temperature values after interpolation: {missing_after}")

    if len(merged) != len(demand):
        raise ValueError("Merge changed the number of demand records.")

    if merged["timestamp_utc"].duplicated().any():
        raise ValueError("Merged dataset contains duplicate UTC timestamps.")

    if missing_after > 0:
        print(
            "\nWARNING: Some temperature values remain missing. "
            "Do not proceed to feature engineering until these gaps "
            "have been investigated."
        )

    merged.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("MERGING COMPLETED")
    print("=" * 65)
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Time range (UTC): {merged['timestamp_utc'].min()} to "
          f"{merged['timestamp_utc'].max()}")
    print(f"Saved dataset: {OUTPUT_FILE}")

    print("\nFirst five records:")
    print(
        merged[
            [
                "timestamp_utc",
                "timestamp_local",
                "Electricity consumption in Finland",
                TEMPERATURE_COLUMN,
            ]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
