
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"

DEMAND_FILE = DATA_DIR / "hourly_electricity_demand.csv"
TEMPERATURE_FILE = DATA_DIR / "hourly_temprature_data.csv"


def inspect_dataset(name: str, file_path: Path, separator: str = ",") -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"{name} dataset was not found: {file_path}"
        )

    data = pd.read_csv(file_path, sep=separator)

    print(f"\n{'=' * 65}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 65}")

    print(f"File: {file_path.name}")
    print(f"Dimensions: {data.shape[0]} rows × {data.shape[1]} columns")

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nData types:")
    print(data.dtypes.to_string())

    print("\nMissing values:")
    missing = data.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    print(f"\nDuplicate rows: {data.duplicated().sum()}")

    print("\nDescriptive statistics:")
    print(data.describe(include="all").transpose().to_string())

    print("\nFirst five records:")
    print(data.head().to_string(index=False))

    return data


def main() -> None:
    demand_data = inspect_dataset(
        "Electricity demand",
        DEMAND_FILE,
        separator=";"
    )

    temperature_data = inspect_dataset(
        "Temperature",
        TEMPERATURE_FILE
    )

    print(f"\n{'=' * 65}")
    print("INITIAL DATA INSPECTION COMPLETED")
    print(f"{'=' * 65}")
    print(f"Demand records: {len(demand_data)}")
    print(f"Temperature records: {len(temperature_data)}")


if __name__ == "__main__":
    main()
