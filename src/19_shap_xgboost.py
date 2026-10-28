
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")

train_data = pd.read_csv(PROCESSED_DIR / "train_data.csv")
test_data = pd.read_csv(PROCESSED_DIR / "test_data.csv")

target = "Electricity consumption in Finland"

features = [
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

X_train = train_data[features].rename(
    columns={"Average temperature [°C]": "temperature"}
)
y_train = train_data[target]

X_test = test_data[features].rename(
    columns={"Average temperature [°C]": "temperature"}
)
y_test = test_data[target]

print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print("Calculating SHAP values...")

sample_size = min(500, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0),
}).sort_values("Mean_Absolute_SHAP", ascending=False)

importance.to_csv(
    RESULTS_DIR / "xgboost_shap_importance.csv",
    index=False,
)

print("\nSHAP feature importance:")
print(importance.to_string(index=False))

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False,
)
plt.savefig(
    RESULTS_DIR / "xgboost_shap_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig(
    RESULTS_DIR / "xgboost_shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

shap.dependence_plot(
    "temperature",
    shap_values,
    X_sample,
    show=False,
)
plt.savefig(
    RESULTS_DIR / "xgboost_temperature_shap_dependence.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

individual_shap = pd.DataFrame(
    shap_values,
    columns=X_sample.columns,
    index=X_sample.index,
)
individual_shap.to_csv(
    RESULTS_DIR / "xgboost_individual_shap_values.csv",
    index_label="test_index",
)

explained_data = X_sample.copy()
explained_data[target] = y_test.loc[X_sample.index]
explained_data.to_csv(
    RESULTS_DIR / "xgboost_shap_explained_test_data.csv",
    index_label="test_index",
)

print(f"\nSHAP analysis completed. Files saved to: {RESULTS_DIR}")
