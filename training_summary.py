import pandas as pd

INPUT = "results/csv/final_precision_report.csv"
OUT = "results/csv/training_dynamic_range_summary.csv"

df = pd.read_csv(INPUT)

# Keep only the ResNet-18 training experiment
train = df[
    df["experiment"] == "ResNet18_CIFAR10_Training"
].copy()

# Find the worst dynamic-range case for each data type
summary = (
    train.loc[
        train.groupby("data_type")["dynamic_range"].idxmax()
    ][
        [
            "data_type",
            "stage",
            "layer",
            "dynamic_range",
            "log10_dynamic_range",
            "maximum_abs",
            "smallest_nonzero",
            "zero_percent"
        ]
    ]
    .sort_values(
        "dynamic_range",
        ascending=False
    )
    .reset_index(drop=True)
)

summary.to_csv(
    OUT,
    index=False
)

print("\n==========================================")
print("TRAINING DYNAMIC-RANGE SUMMARY")
print("==========================================\n")

print(summary.to_string(index=False))

print("\nSaved to:")
print(OUT)

# ------------------------------------------------------------
# Additional summary by stage
# ------------------------------------------------------------

stage_summary = (
    train.groupby(
        ["data_type", "stage"]
    )["dynamic_range"]
    .max()
    .reset_index()
    .sort_values(
        ["data_type", "dynamic_range"],
        ascending=[True, False]
    )
)

stage_summary.to_csv(
    "results/csv/training_dynamic_range_by_stage.csv",
    index=False
)

print("\n==========================================")
print("WORST DYNAMIC RANGE BY STAGE")
print("==========================================\n")

print(stage_summary.to_string(index=False))