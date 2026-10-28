


from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. CONFIGURATION AND PROJECT PATHS
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
SAMPLE_SIZE = 500

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

TRAIN_PATH = PROCESSED_DIR / "train_data.csv"
TEST_PATH = PROCESSED_DIR / "test_data.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "Electricity consumption in Finland"
TEMPERATURE_FEATURE = "Average temperature [°C]"

FEATURE_COLUMNS = [
    "Average temperature [°C]",
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("\nLoading training and test data...")

for path in [TRAIN_PATH, TEST_PATH]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset not found: {path}")

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

for name, data in [
    ("Training", train_data),
    ("Test", test_data),
]:
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} data is missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# 3. PREPARE FEATURES AND TARGET
# ============================================================

X_train = train_data[FEATURE_COLUMNS]
y_train = train_data[TARGET_COLUMN]

X_test = test_data[FEATURE_COLUMNS]
y_test = test_data[TARGET_COLUMN]

print(f"Number of features: {len(FEATURE_COLUMNS)}")


# ============================================================
# 4. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

random_forest.fit(X_train, y_train)

print("Random Forest training completed.")


# ============================================================
# 5. SELECT TEST OBSERVATIONS FOR SHAP
# ============================================================

print("\nSelecting test observations for SHAP analysis...")

sample_size = min(SAMPLE_SIZE, len(X_test))

X_explain = X_test.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

print(f"Number of observations explained: {len(X_explain)}")


# ============================================================
# 6. CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(random_forest)
shap_values = explainer.shap_values(X_explain)

# Compatibility with SHAP versions returning a list.
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

# Some SHAP versions may include a trailing output dimension.
if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values[:, :, 0]

if shap_values.shape != X_explain.shape:
    raise ValueError(
        f"Unexpected SHAP shape {shap_values.shape}; "
        f"expected {X_explain.shape}."
    )

print("SHAP value calculation completed.")
print(f"SHAP values shape: {shap_values.shape}")


# ============================================================
# 7. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

print("\nCalculating global SHAP feature importance...")

mean_absolute_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Mean_Absolute_SHAP": mean_absolute_shap,
}).sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False,
)

shap_importance_path = (
    RESULTS_DIR / "random_forest_shap_importance.csv"
)

shap_importance.to_csv(shap_importance_path, index=False)

print("\nGlobal SHAP feature importance:")
print(shap_importance.to_string(index=False))
print(f"\nSaved SHAP importance CSV:\n{shap_importance_path}")


# ============================================================
# 8. GLOBAL SHAP FEATURE IMPORTANCE BAR PLOT
# ============================================================

print("\nCreating SHAP feature importance bar plot...")

importance_sorted = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=True,
)

plt.figure(figsize=(10, 7))
plt.barh(
    importance_sorted["Feature"],
    importance_sorted["Mean_Absolute_SHAP"],
)
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Random Forest Global SHAP Feature Importance")
plt.tight_layout()

bar_plot_path = RESULTS_DIR / "random_forest_shap_importance.png"
plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved bar plot:\n{bar_plot_path}")


# ============================================================
# 9. SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_explain,
    show=False,
    plot_size=(11, 8),
)

plt.title("Random Forest SHAP Summary Plot")
plt.tight_layout()

summary_plot_path = RESULTS_DIR / "random_forest_shap_summary.png"
plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved summary plot:\n{summary_plot_path}")


# ============================================================
# 10. TEMPERATURE SHAP DEPENDENCE PLOT
# ============================================================

print("\nCreating temperature SHAP dependence plot...")

temperature_feature_index = FEATURE_COLUMNS.index(
    TEMPERATURE_FEATURE
)

plt.figure(figsize=(10, 7))

shap.dependence_plot(
    temperature_feature_index,
    shap_values,
    X_explain,
    interaction_index=None,
    show=False,
)

plt.title(
    "SHAP Dependence Plot: Temperature "
    "in Random Forest Forecasting"
)
plt.tight_layout()

temperature_plot_path = (
    RESULTS_DIR / "random_forest_temperature_shap_dependence.png"
)

plt.savefig(
    temperature_plot_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(f"Saved temperature dependence plot:\n{temperature_plot_path}")


# ============================================================
# 11. SAVE INDIVIDUAL SHAP VALUES
# ============================================================

print("\nSaving individual SHAP values...")

shap_values_dataframe = pd.DataFrame(
    shap_values,
    columns=FEATURE_COLUMNS,
)

shap_values_path = (
    RESULTS_DIR / "random_forest_individual_shap_values.csv"
)

shap_values_dataframe.to_csv(shap_values_path, index=False)

print(f"Saved individual SHAP values:\n{shap_values_path}")


# ============================================================
# 12. SAVE EXPLAINED TEST DATA
# ============================================================

explained_test_data_path = (
    RESULTS_DIR / "random_forest_shap_explained_test_data.csv"
)

X_explain.to_csv(explained_test_data_path, index=False)

print(f"Saved explained test data:\n{explained_test_data_path}")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SHAP RANDOM FOREST ANALYSIS COMPLETED")
print("=" * 60)

print("\nGenerated files:")
for path in [
    shap_importance_path,
    bar_plot_path,
    summary_plot_path,
    temperature_plot_path,
    shap_values_path,
    explained_test_data_path,
]:
    print(f"- {path}")

print("\nAll SHAP files were saved in the results folder.")
