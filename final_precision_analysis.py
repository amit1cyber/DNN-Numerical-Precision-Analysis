import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CSV_DIR = "results/csv"
OUT = f"{CSV_DIR}/final_precision_analysis.csv"

os.makedirs(CSV_DIR, exist_ok=True)


# ============================================================
# FORMAT CHARACTERISTICS
# ============================================================

# Approximate finite ranges used for suitability analysis.
FP16_MAX = 65504.0

# FP8 formats:
# E4M3: approximately +/-448
# E5M2: approximately +/-57344
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0

INT8_MAX = 127
INT16_MAX = 32767


# ============================================================
# LOAD ALL EXPERIMENT CSV FILES
# ============================================================

files = glob.glob(
    os.path.join(CSV_DIR, "*.csv")
)

rows = []


for filename in files:

    # Don't read this output back into itself
    if filename.endswith(
        "final_precision_analysis.csv"
    ):
        continue

    try:
        df = pd.read_csv(filename)
    except Exception:
        continue

    required = {
        "layer",
        "minimum",
        "maximum",
        "smallest_nonzero",
        "zero_percent",
        "minimum_spacing",
        "maximum_abs"
    }

    if not required.issubset(df.columns):
        continue

    base = os.path.basename(filename)

    # Remove .csv
    name = base[:-4]

    # Identify experiment
    if name.startswith("training_"):
        parts = name.split("_")

        # training_beginning_weights
        stage = parts[1]
        data_type = "_".join(parts[2:])

        experiment = "ResNet18_CIFAR10_Training"

    elif name.startswith("resnet18_cifar10_"):

        stage = "inference"
        data_type = name[
            len("resnet18_cifar10_"):
        ]

        experiment = "ResNet18_CIFAR10_Inference"

    elif name.startswith(
        "mobilenetv2_cifar100_"
    ):

        stage = "inference"
        data_type = name[
            len("mobilenetv2_cifar100_"):
        ]

        experiment = "MobileNetV2_CIFAR100"

    elif name.startswith("lenet5_mnist_"):

        stage = "inference"
        data_type = name[
            len("lenet5_mnist_"):
        ]

        experiment = "LeNet5_MNIST"

    else:
        continue

    for _, r in df.iterrows():

        min_v = float(r["minimum"])
        max_v = float(r["maximum"])
        absmax = float(r["maximum_abs"])

        smallest = float(
            r["smallest_nonzero"]
        )

        spacing = float(
            r["minimum_spacing"]
        )

        zero_pct = float(
            r["zero_percent"]
        )

        # ====================================================
        # RANGE
        # ====================================================

        dynamic_range = (
            absmax / smallest
            if smallest > 0
            else np.inf
        )

        # ====================================================
        # FP16
        # ====================================================

        fp16_range_ok = (
            absmax <= FP16_MAX
        )

        # ====================================================
        # FP8
        # ====================================================

        fp8_e4m3_range_ok = (
            absmax <= FP8_E4M3_MAX
        )

        fp8_e5m2_range_ok = (
            absmax <= FP8_E5M2_MAX
        )

        # ====================================================
        # INTEGER SCALED REPRESENTATION
        # ====================================================

        # Symmetric per-tensor scaling.
        #
        # scale = max(abs(x)) / maximum_integer
        #
        # Quantization step is equal to scale.
        # ====================================================

        if absmax > 0:

            int8_step = (
                absmax / INT8_MAX
            )

            int16_step = (
                absmax / INT16_MAX
            )

            # Values below half a quantization step
            # can round to zero.
            int8_zero_threshold = (
                int8_step / 2
            )

            int16_zero_threshold = (
                int16_step / 2
            )

            int8_small_values_lost = (
                smallest <
                int8_zero_threshold
            )

            int16_small_values_lost = (
                smallest <
                int16_zero_threshold
            )

        else:

            int8_step = 0
            int16_step = 0

            int8_zero_threshold = 0
            int16_zero_threshold = 0

            int8_small_values_lost = False
            int16_small_values_lost = False

        # ====================================================
        # SIMPLE SUITABILITY CLASSIFICATION
        # ====================================================

        if not fp16_range_ok:

            fp16_result = "NOT REPRESENTABLE"

        else:

            fp16_result = "RANGE ADEQUATE"

        if not fp8_e4m3_range_ok:

            fp8_e4m3_result = (
                "RANGE EXCEEDED"
            )

        else:

            fp8_e4m3_result = (
                "RANGE ADEQUATE"
            )

        if not fp8_e5m2_range_ok:

            fp8_e5m2_result = (
                "RANGE EXCEEDED"
            )

        else:

            fp8_e5m2_result = (
                "RANGE ADEQUATE"
            )

        if int8_small_values_lost:

            int8_result = (
                "SMALL VALUES MAY BE LOST"
            )

        else:

            int8_result = (
                "RANGE/STEP APPEARS ADEQUATE"
            )

        if int16_small_values_lost:

            int16_result = (
                "SMALL VALUES MAY BE LOST"
            )

        else:

            int16_result = (
                "RANGE/STEP APPEARS ADEQUATE"
            )

        rows.append({

            "experiment":
                experiment,

            "stage":
                stage,

            "data_type":
                data_type,

            "layer":
                r["layer"],

            "minimum":
                min_v,

            "maximum":
                max_v,

            "maximum_abs":
                absmax,

            "smallest_nonzero":
                smallest,

            "zero_percent":
                zero_pct,

            "observed_min_spacing":
                spacing,

            "dynamic_range":
                dynamic_range,

            "FP16_range_ok":
                fp16_range_ok,

            "FP16_result":
                fp16_result,

            "FP8_E4M3_range_ok":
                fp8_e4m3_range_ok,

            "FP8_E4M3_result":
                fp8_e4m3_result,

            "FP8_E5M2_range_ok":
                fp8_e5m2_range_ok,

            "FP8_E5M2_result":
                fp8_e5m2_result,

            "INT8_scale":
                int8_step,

            "INT8_quantization_step":
                int8_step,

            "INT8_zero_threshold":
                int8_zero_threshold,

            "INT8_smallest_may_be_lost":
                int8_small_values_lost,

            "INT8_result":
                int8_result,

            "INT16_scale":
                int16_step,

            "INT16_quantization_step":
                int16_step,

            "INT16_zero_threshold":
                int16_zero_threshold,

            "INT16_smallest_may_be_lost":
                int16_small_values_lost,

            "INT16_result":
                int16_result
        })


# ============================================================
# SAVE FINAL TABLE
# ============================================================

result = pd.DataFrame(rows)

result = result.sort_values(
    [
        "experiment",
        "stage",
        "data_type",
        "layer"
    ]
)

result.to_csv(
    OUT,
    index=False
)

print()
print("==========================================")
print("FINAL PRECISION ANALYSIS COMPLETE")
print("==========================================")
print()
print("Output:")
print(OUT)
print()
print("Rows analyzed:", len(result))
print()