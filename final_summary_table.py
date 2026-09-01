import pandas as pd
import numpy as np
import os
INPUT = "results/csv/final_precision_report.csv"

df = pd.read_csv(INPUT)

# ------------------------------------------------------------
# 1. TRAINING SUMMARY
# ------------------------------------------------------------

train = df[
    df["experiment"] == "ResNet18_CIFAR10_Training"
].copy()

# Worst case = largest observed dynamic range
idx = train.groupby("data_type")["dynamic_range"].idxmax()

summary = train.loc[idx].copy()

# Keep only the requested categories
order = [
    "inputs",
    "weights",
    "activations",
    "gradients",
    "updates",
    "biases"
]

summary["order"] = summary["data_type"].apply(
    lambda x: order.index(x) if x in order else 999
)

summary = summary.sort_values("order")


# ------------------------------------------------------------
# 2. SIMPLE FORMAT SUITABILITY
# ------------------------------------------------------------

def fp16_label(row):
    if not row["FP16_range_ok"]:
        return "No"
    if row["FP16_underflow_possible"]:
        return "Caution"
    return "Likely"

def fp8_label(row, prefix):
    if not row[f"{prefix}_range_ok"]:
        return "No"
    if not row[f"{prefix}_subnormal_ok"]:
        return "Caution"
    return "Range OK"

def integer_label(row, bits):
    col = f"INT{bits}_smallest_may_be_lost"
    return "Caution" if row[col] else "Likely"


summary["FP16"] = summary.apply(
    fp16_label,
    axis=1
)

summary["FP8_E4M3"] = summary.apply(
    lambda r: fp8_label(r, "FP8_E4M3"),
    axis=1
)

summary["FP8_E5M2"] = summary.apply(
    lambda r: fp8_label(r, "FP8_E5M2"),
    axis=1
)

summary["INT16"] = summary.apply(
    lambda r: integer_label(r, 16),
    axis=1
)

summary["INT8"] = summary.apply(
    lambda r: integer_label(r, 8),
    axis=1
)


# ------------------------------------------------------------
# 3. FINAL CONCLUSION COLUMN
# ------------------------------------------------------------

def conclusion(row):

    dtype = row["data_type"]

    if dtype == "inputs":
        return (
            "FP16 is strongly plausible; "
            "INT8 requires scaling/calibration."
        )

    if dtype == "weights":
        return (
            "FP16 is a strong candidate; "
            "INT8/INT16 may be possible with suitable scaling."
        )

    if dtype == "activations":
        return (
            "FP16 is a strong candidate; "
            "FP8 may be viable for selected layers."
        )

    if dtype == "gradients":
        return (
            "Higher precision is preferable; "
            "FP8 and fixed-point formats are precision-sensitive."
        )

    if dtype == "updates":
        return (
            "Most precision-sensitive quantity; "
            "FP32 is safest and FP16 may lose tiny updates."
        )

    if dtype == "biases":
        return (
            "FP16 is a strong candidate; "
            "integer formats require appropriate scaling."
        )

    return ""


summary["overall_conclusion"] = summary.apply(
    conclusion,
    axis=1
)


# ------------------------------------------------------------
# 4. SELECT REPORT COLUMNS
# ------------------------------------------------------------

report = summary[
    [
        "data_type",
        "stage",
        "layer",
        "maximum_abs",
        "smallest_nonzero",
        "dynamic_range",
        "log10_dynamic_range",
        "FP16",
        "FP8_E4M3",
        "FP8_E5M2",
        "INT16",
        "INT8",
        "overall_conclusion"
    ]
].copy()


# ------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------

output = (
    "results/csv/"
    "final_training_precision_summary.csv"
)

report.to_csv(
    output,
    index=False
)


# ------------------------------------------------------------
# 6. HUMAN-READABLE PRINT
# ------------------------------------------------------------

print("\n==========================================================")
print("FINAL TRAINING PRECISION SUMMARY")
print("==========================================================\n")

for _, r in report.iterrows():

    print(
        f"{r['data_type'].upper():12s} | "
        f"Worst layer: {r['layer']} | "
        f"Stage: {r['stage']} | "
        f"Dynamic range: {r['dynamic_range']:.3e} | "
        f"Orders: {r['log10_dynamic_range']:.2f}"
    )

    print(
        f"             FP16={r['FP16']} | "
        f"E4M3={r['FP8_E4M3']} | "
        f"E5M2={r['FP8_E5M2']} | "
        f"INT16={r['INT16']} | "
        f"INT8={r['INT8']}"
    )

    print(
        f"             {r['overall_conclusion']}"
    )

    print()


print("Saved:")
print(output)
print()