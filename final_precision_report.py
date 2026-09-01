import os
import glob
import math
import numpy as np
import pandas as pd

CSV_DIR = "results/csv"
OUTPUT = os.path.join(
    CSV_DIR, "final_precision_report.csv"
)

# ============================================================
# FORMAT LIMITS
# ============================================================

# FP16
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2 ** -14
FP16_MIN_SUBNORMAL = 2 ** -24
FP16_FRAC_BITS = 10

# FP8 E4M3
FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN_NORMAL = 2 ** -6
FP8_E4M3_MIN_SUBNORMAL = 2 ** -9
FP8_E4M3_FRAC_BITS = 3

# FP8 E5M2
FP8_E5M2_MAX = 57344.0
FP8_E5M2_MIN_NORMAL = 2 ** -14
FP8_E5M2_MIN_SUBNORMAL = 2 ** -16
FP8_E5M2_FRAC_BITS = 2

INT8_MAX = 127
INT16_MAX = 32767


# ============================================================
# FLOATING-POINT SPACING
# ============================================================

def approx_spacing(x, frac_bits):
    """
    Approximate spacing of a normalized floating-point
    representation near magnitude x.
    """
    x = abs(float(x))

    if x == 0:
        return 0.0

    exponent = math.floor(math.log2(x))

    return 2 ** (exponent - frac_bits)


# ============================================================
# LOAD CSV DATA
# ============================================================

rows = []

for filename in glob.glob(
    os.path.join(CSV_DIR, "*.csv")
):

    if filename.endswith(
        "final_precision_report.csv"
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

    file = os.path.basename(filename)
    name = file[:-4]

    # --------------------------------------------------------
    # Identify experiment / stage / data type
    # --------------------------------------------------------

    if name.startswith("training_"):

        parts = name.split("_")

        experiment = "ResNet18_CIFAR10_Training"
        stage = parts[1]
        data_type = "_".join(parts[2:])

    elif name.startswith("resnet18_cifar10_"):

        experiment = "ResNet18_CIFAR10_Inference"
        stage = "inference"
        data_type = name[
            len("resnet18_cifar10_"):
        ]

    elif name.startswith("mobilenetv2_cifar100_"):

        experiment = "MobileNetV2_CIFAR100"
        stage = "inference"
        data_type = name[
            len("mobilenetv2_cifar100_"):
        ]

    elif name.startswith("lenet5_mnist_"):

        experiment = "LeNet5_MNIST"
        stage = "inference"
        data_type = name[
            len("lenet5_mnist_"):
        ]

    else:
        continue

    # --------------------------------------------------------
    # Analyze every layer
    # --------------------------------------------------------

    for _, r in df.iterrows():

        minimum = float(r["minimum"])
        maximum = float(r["maximum"])
        max_abs = float(r["maximum_abs"])
        smallest = float(r["smallest_nonzero"])
        zero_pct = float(r["zero_percent"])
        observed_spacing = float(
            r["minimum_spacing"]
        )

        # ----------------------------------------------------
        # Dynamic range
        # ----------------------------------------------------

        if smallest > 0:
            dynamic_range = max_abs / smallest
            log_dynamic_range = math.log10(
                dynamic_range
            )
        else:
            dynamic_range = np.inf
            log_dynamic_range = np.inf

        # ====================================================
        # FP16
        # ====================================================

        fp16_range_ok = (
            max_abs <= FP16_MAX
        )

        fp16_underflow_possible = (
            smallest < FP16_MIN_SUBNORMAL
        )

        fp16_min_spacing_at_max = (
            approx_spacing(
                max_abs,
                FP16_FRAC_BITS
            )
            if max_abs > 0
            else 0
        )

        # ====================================================
        # FP8 E4M3
        # ====================================================

        e4m3_range_ok = (
            max_abs <= FP8_E4M3_MAX
        )

        e4m3_subnormal_ok = (
            smallest >= FP8_E4M3_MIN_SUBNORMAL
        )

        e4m3_spacing_at_max = (
            approx_spacing(
                max_abs,
                FP8_E4M3_FRAC_BITS
            )
            if max_abs > 0
            else 0
        )

        # ====================================================
        # FP8 E5M2
        # ====================================================

        e5m2_range_ok = (
            max_abs <= FP8_E5M2_MAX
        )

        e5m2_subnormal_ok = (
            smallest >= FP8_E5M2_MIN_SUBNORMAL
        )

        e5m2_spacing_at_max = (
            approx_spacing(
                max_abs,
                FP8_E5M2_FRAC_BITS
            )
            if max_abs > 0
            else 0
        )

        # ====================================================
        # INT8
        # ====================================================

        if max_abs > 0:

            int8_scale = (
                max_abs / INT8_MAX
            )

            int8_half_step = (
                int8_scale / 2
            )

            int8_smallest_lost = (
                smallest < int8_half_step
            )

            # Rough fraction of observed values that
            # fall inside one half-step cannot be known
            # from summary statistics alone.
            int8_resolution_ratio = (
                smallest / int8_scale
            )

        else:

            int8_scale = 0
            int8_half_step = 0
            int8_smallest_lost = False
            int8_resolution_ratio = 0

        # ====================================================
        # INT16
        # ====================================================

        if max_abs > 0:

            int16_scale = (
                max_abs / INT16_MAX
            )

            int16_half_step = (
                int16_scale / 2
            )

            int16_smallest_lost = (
                smallest < int16_half_step
            )

            int16_resolution_ratio = (
                smallest / int16_scale
            )

        else:

            int16_scale = 0
            int16_half_step = 0
            int16_smallest_lost = False
            int16_resolution_ratio = 0

        # ====================================================
        # QUALITATIVE RECOMMENDATIONS
        # ====================================================

        # FP16
        if not fp16_range_ok:
            fp16_recommendation = "NOT SUITABLE: RANGE"
        elif fp16_underflow_possible:
            fp16_recommendation = (
                "CAUTION: VERY SMALL VALUES"
            )
        else:
            fp16_recommendation = "LIKELY SUITABLE"

        # E4M3
        if not e4m3_range_ok:
            e4m3_recommendation = (
                "NOT SUITABLE: RANGE"
            )
        elif not e4m3_subnormal_ok:
            e4m3_recommendation = (
                "CAUTION: VERY SMALL VALUES"
            )
        else:
            e4m3_recommendation = "RANGE SUITABLE"

        # E5M2
        if not e5m2_range_ok:
            e5m2_recommendation = (
                "NOT SUITABLE: RANGE"
            )
        elif not e5m2_subnormal_ok:
            e5m2_recommendation = (
                "CAUTION: VERY SMALL VALUES"
            )
        else:
            e5m2_recommendation = "RANGE SUITABLE"

        # INT8
        if int8_smallest_lost:
            int8_recommendation = (
                "CAUTION: SMALL VALUES MAY ROUND TO ZERO"
            )
        else:
            int8_recommendation = (
                "RESOLUTION APPEARS ADEQUATE"
            )

        # INT16
        if int16_smallest_lost:
            int16_recommendation = (
                "CAUTION: SMALL VALUES MAY ROUND TO ZERO"
            )
        else:
            int16_recommendation = (
                "RESOLUTION APPEARS ADEQUATE"
            )

        rows.append({

            "experiment": experiment,
            "stage": stage,
            "data_type": data_type,
            "layer": r["layer"],

            "minimum": minimum,
            "maximum": maximum,
            "maximum_abs": max_abs,
            "smallest_nonzero": smallest,
            "zero_percent": zero_pct,
            "observed_min_spacing": observed_spacing,

            "dynamic_range": dynamic_range,
            "log10_dynamic_range":
                log_dynamic_range,

            # FP16
            "FP16_range_ok":
                fp16_range_ok,
            "FP16_underflow_possible":
                fp16_underflow_possible,
            "FP16_spacing_at_max":
                fp16_min_spacing_at_max,
            "FP16_recommendation":
                fp16_recommendation,

            # FP8 E4M3
            "FP8_E4M3_range_ok":
                e4m3_range_ok,
            "FP8_E4M3_smallest_representable":
                FP8_E4M3_MIN_SUBNORMAL,
            "FP8_E4M3_subnormal_ok":
                e4m3_subnormal_ok,
            "FP8_E4M3_spacing_at_max":
                e4m3_spacing_at_max,
            "FP8_E4M3_recommendation":
                e4m3_recommendation,

            # FP8 E5M2
            "FP8_E5M2_range_ok":
                e5m2_range_ok,
            "FP8_E5M2_smallest_representable":
                FP8_E5M2_MIN_SUBNORMAL,
            "FP8_E5M2_subnormal_ok":
                e5m2_subnormal_ok,
            "FP8_E5M2_spacing_at_max":
                e5m2_spacing_at_max,
            "FP8_E5M2_recommendation":
                e5m2_recommendation,

            # INT8
            "INT8_scale":
                int8_scale,
            "INT8_step":
                int8_scale,
            "INT8_half_step":
                int8_half_step,
            "INT8_smallest_to_step_ratio":
                int8_resolution_ratio,
            "INT8_smallest_may_be_lost":
                int8_smallest_lost,
            "INT8_recommendation":
                int8_recommendation,

            # INT16
            "INT16_scale":
                int16_scale,
            "INT16_step":
                int16_scale,
            "INT16_half_step":
                int16_half_step,
            "INT16_smallest_to_step_ratio":
                int16_resolution_ratio,
            "INT16_smallest_may_be_lost":
                int16_smallest_lost,
            "INT16_recommendation":
                int16_recommendation
        })


# ============================================================
# SAVE
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
    OUTPUT,
    index=False
)

print()
print("============================================")
print("FINAL PRECISION REPORT GENERATED")
print("============================================")
print(f"Rows analyzed: {len(result)}")
print()
print("Output:")
print(OUTPUT)
print()
print("No model was trained or executed.")
print("Only existing FP32 statistics were analyzed.")
print("============================================")