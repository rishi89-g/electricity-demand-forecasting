
import os
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Model result files
# ============================================================

MODEL_FILES = {
    "Naive Forecast": "naive_results.csv",
    "Linear Regression": "linear_regression_results.csv",
    "Random Forest": "random_forest_results.csv",
    "XGBoost": "xgboost_results.csv",
    "LSTM": "lstm_results.csv",
}


# ============================================================
# 1. Read and normalize result files
# ============================================================

all_results = []

for model_name, filename in MODEL_FILES.items():

    file_path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Format A: Separate validation/test columns
    if {
        "validation_mae",
        "validation_rmse",
        "test_mae",
        "test_rmse"
    }.issubset(df.columns):

        row = df.iloc[0]

        all_results.append({
            "Model": model_name,
            "Validation_MAE": row["validation_mae"],
            "Validation_RMSE": row["validation_rmse"],
            "Test_MAE": row["test_mae"],
            "Test_RMSE": row["test_rmse"]
        })

    # Format B: Dataset rows (Naive Forecast)
    elif {
        "dataset",
        "mae",
        "rmse"
    }.issubset(df.columns):

        validation = df[
            df["dataset"].str.lower() == "validation"
        ]

        test = df[
            df["dataset"].str.lower() == "test"
        ]

        if validation.empty or test.empty:
            raise ValueError(
                f"Validation or Test results missing for {model_name}"
            )

        all_results.append({
            "Model": model_name,
            "Validation_MAE": validation.iloc[0]["mae"],
            "Validation_RMSE": validation.iloc[0]["rmse"],
            "Test_MAE": test.iloc[0]["mae"],
            "Test_RMSE": test.iloc[0]["rmse"]
        })

    else:
        raise ValueError(
            f"Unrecognized format in {filename}.\n"
            f"Columns found: {list(df.columns)}"
        )


# ============================================================
# 2. Create comparison table
# ============================================================

comparison = pd.DataFrame(all_results)

print("\n========================================")
print("MODEL PERFORMANCE COMPARISON")
print("========================================")

print(comparison.to_string(index=False))


# ============================================================
# 3. Save comparison table
# ============================================================

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_summary.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nComparison table saved to:")
print(comparison_path)


# ============================================================
# 4. Prepare chart data
# ============================================================

models = comparison["Model"].tolist()

validation_mae = comparison["Validation_MAE"].tolist()
test_mae = comparison["Test_MAE"].tolist()

validation_rmse = comparison["Validation_RMSE"].tolist()
test_rmse = comparison["Test_RMSE"].tolist()


# ============================================================
# 5. MAE comparison chart
# ============================================================

x = range(len(models))
width = 0.35

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    validation_mae,
    width=width,
    label="Validation MAE"
)

plt.bar(
    [i + width / 2 for i in x],
    test_mae,
    width=width,
    label="Test MAE"
)

plt.title("Model Comparison: MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(list(x), models, rotation=30, ha="right")
plt.legend()

plt.tight_layout()

mae_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_mae.png"
)

plt.savefig(mae_path, dpi=300)
plt.close()

print("MAE chart saved to:")
print(mae_path)


# ============================================================
# 6. RMSE comparison chart
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width / 2 for i in x],
    validation_rmse,
    width=width,
    label="Validation RMSE"
)

plt.bar(
    [i + width / 2 for i in x],
    test_rmse,
    width=width,
    label="Test RMSE"
)

plt.title("Model Comparison: RMSE")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(list(x), models, rotation=30, ha="right")
plt.legend()

plt.tight_layout()

rmse_path = os.path.join(
    RESULTS_DIR,
    "model_comparison_rmse.png"
)

plt.savefig(rmse_path, dpi=300)
plt.close()

print("RMSE chart saved to:")
print(rmse_path)


