DNN NUMERICAL RANGE AND PRECISION ANALYSIS
CONCLUSION AND PRECISION COMPARISON MATRIX

1. OVERALL FINDINGS

The numerical analysis of ResNet-18 training on CIFAR-10 shows that different
DNN quantities have substantially different numerical ranges and precision
requirements. The largest dynamic ranges were observed for gradients and
weight updates, while biases exhibited the smallest dynamic range.

Data Type           Worst Dynamic Range      Orders of Magnitude
-----------------------------------------------------------------
Gradients            3.36 x 10^11             11.53
Weight updates       3.16 x 10^10             10.50
Weights              2.33 x 10^9              9.37
Activations          4.55 x 10^7              7.66
Inputs               3.38 x 10^7              7.53
Biases               4.79 x 10^4              4.68

The results show that training quantities, particularly gradients and weight
updates, are considerably more precision-sensitive than many forward-path
quantities.


2. FP32 VS LOWER-PRECISION COMPARISON MATRIX

Legend:
  ++  Strong candidate based on numerical analysis
  +   Potentially adequate
  +/- Conditional; requires careful scaling or precision management
  -   Not suitable for preserving the complete observed range under the
      analyzed conditions

DNN Data          FP32   FP16   FP8 E4M3   FP8 E5M2   INT16   INT8
---------------------------------------------------------------------
Inputs             ++     ++       +/-        +/-       +/-     +/-
Weights            ++     ++       +/-        +/-       +/-     +/-
Activations        ++     ++       +/-        +/-       +/-     +/-
Biases             ++     ++       +/-        +/-        +      +/-
Gradients          ++     +/-       -         +/-        -       -
Weight updates     ++     +/-       -          -         -       -
Output logits      ++     +        +/-        +/-       +/-     +/-


3. FORMAT-WISE COMPARISON

Criterion                  FP32      FP16      FP8       INT16      INT8
---------------------------------------------------------------------------
Numerical range            Excellent High      Moderate  Scaled     Scaled
Precision                  Highest   High      Lower     Fixed      Fixed
Small-value preservation   Excellent Good*     Limited   Limited*   Limited*
Weights                    Excellent Strong    Conditional Conditional Conditional
Activations                Excellent Strong    Conditional Conditional Conditional
Gradients                  Strongest Conditional Poor/Conditional Poor      Poor
Updates                    Strongest Conditional Poor      Poor      Poor

* Depending on the observed magnitude distribution and whether the smallest
  values are important to the computation.


4. KEY NUMERICAL EVIDENCE

4.1 Gradients

The worst observed gradient tensor was:

  Layer:
    layer4.0.downsample.0.weight

  Maximum absolute magnitude:
    1.7846 x 10^-2

  Smallest observed nonzero magnitude:
    5.3142 x 10^-14

  Dynamic range:
    3.36 x 10^11

  Log10(dynamic range):
    11.53

Interpretation:
The largest and smallest observed gradient values differ by more than
eleven orders of magnitude. This makes gradients one of the most
precision-sensitive data types in the training process. A low-precision
representation with coarse resolution can lose small gradient components
even though the overall gradient tensor remains within the format's range.

Implication:
Higher precision is preferable for gradients. FP16 may be usable with
appropriate mixed-precision techniques, while direct FP8 or fixed-point
representation with one scale is substantially more challenging.


4.2 Weight Updates

The worst observed update tensor was:

  Layer:
    layer4.0.conv2.weight

  Maximum absolute magnitude:
    1.1071 x 10^-3

  Smallest observed nonzero magnitude:
    3.5034 x 10^-14

  Dynamic range:
    3.16 x 10^10

  Log10(dynamic range):
    10.50

Interpretation:
Weight updates span more than ten orders of magnitude. Some updates are
extremely small compared with the largest updates.

Implication:
Weight updates are among the most precision-sensitive quantities in
training. FP32 is the safest representation. FP16 can be attractive in
mixed-precision systems but cannot preserve every very small observed
update. INT8 and INT16 with a single per-tensor scale can also lose the
smallest updates.


4.3 Weights

The worst observed weight tensor was:

  Layer:
    layer4.0.conv2.weight

  Maximum absolute magnitude:
    1.6217 x 10^-1

  Smallest observed nonzero magnitude:
    6.9718 x 10^-11

  Dynamic range:
    2.33 x 10^9

  Log10(dynamic range):
    9.37

Interpretation:
Weights have a large dynamic range, but their numerical magnitudes are
generally much easier to represent than gradients and updates.

Implication:
FP16 is a strong candidate for stored weights. INT8 or INT16 may also be
possible when appropriate scaling, calibration, and possibly per-channel
quantization are used.


4.4 Activations

The worst observed activation tensor was:

  Layer:
    layer4.1.conv2

  Maximum absolute magnitude:
    3.0117 x 10^-1

  Smallest observed nonzero magnitude:
    6.6211 x 10^-9

  Dynamic range:
    4.55 x 10^7

  Log10(dynamic range):
    7.66

Interpretation:
Activations have a substantial dynamic range but are less extreme than
gradients and weight updates.

Implication:
FP16 is a strong candidate for many activation tensors. FP8 can be
considered for selected layers, but its reduced precision near zero must be
examined carefully.


4.5 Inputs

The worst observed input-related tensor was:

  Layer:
    layer1.0.conv1

  Maximum absolute magnitude:
    9.5788

  Smallest observed nonzero magnitude:
    2.8312 x 10^-7

  Dynamic range:
    3.38 x 10^7

  Log10(dynamic range):
    7.53

  Zero percentage:
    33.8%

Interpretation:
The input-related distribution contains a significant number of zeros.
Zero itself is exactly representable in FP16, FP8, INT16, and INT8. The
precision challenge is therefore mainly the preservation of small nonzero
values.

Implication:
FP16 is a strong candidate. Integer representations may also be possible
when proper scaling and calibration are used.


4.6 Biases

The worst observed bias tensor was:

  Layer:
    layer4.1.bn2.bias

  Maximum absolute magnitude:
    7.9016 x 10^-2

  Smallest observed nonzero magnitude:
    1.6510 x 10^-6

  Dynamic range:
    4.79 x 10^4

  Log10(dynamic range):
    4.68

Interpretation:
Biases exhibited the smallest worst-case dynamic range among the major
training quantities.

Implication:
Biases are comparatively tolerant of reduced precision. FP16 is a strong
candidate, and INT16 may also be feasible with suitable scaling.


5. INT8 AND INT16 SCALING ANALYSIS

Symmetric per-tensor scaling was evaluated using:

  S = max(|x|) / Qmax

where:

  Qmax = 127       for INT8
  Qmax = 32767     for INT16

The quantization step is equal to the scale S.

For many tensors, especially gradients and weight updates, the smallest
observed values are much smaller than half of the corresponding quantization
step. These values can therefore round to zero.

This does not mean that INT8 or INT16 DNNs are inherently unsuitable.
Practical quantized systems may use per-channel scaling, clipping,
calibration, mixed precision, stochastic rounding, or other techniques.
The present assignment evaluates numerical range and precision rather than
end-to-end quantized model accuracy.


6. FINAL PRECISION RECOMMENDATION

DNN Data          Recommended Representation       Main Reason
---------------------------------------------------------------------------
Inputs             FP16                            Good range/precision tradeoff
Weights            FP16                            Strong candidate for storage
Activations        FP16                            Strong forward-path candidate
Biases              FP16 / INT16                   Comparatively small range
Gradients           FP32 or carefully managed FP16  Very large dynamic range
Weight updates      FP32 or carefully managed FP16  Extremely small updates
Output logits       FP16                            Generally suitable by range

FP8 can be considered for selected forward-path quantities, but should be
treated as layer-dependent because of its reduced precision.

INT8 provides the largest reduction in numerical representation size among
the examined formats but is most sensitive to scaling and the presence of
very small values. INT16 provides finer resolution than INT8 but still cannot
preserve the full observed dynamic range of some training quantities using a
single scale.


7. OVERALL CONCLUSION

The experiment demonstrates that there is no single numerical representation
that is optimal for every DNN quantity.

The observed dynamic ranges vary from approximately 4.79 x 10^4 for the
worst-case bias to 3.36 x 10^11 for the worst-case gradient. This difference
shows why precision requirements must be considered separately for each
type of data and, ideally, for individual layers.

FP16 emerges as the most generally promising lower-precision alternative to
FP32 for many forward-path quantities, including inputs, weights,
activations, biases, and output logits. It provides a substantially better
precision/range trade-off than FP8 and avoids the coarse fixed-step behavior
of integer formats.

FP8 may be appropriate for selected forward computations, but the observed
small-valued gradients and updates demonstrate that it cannot preserve all
of the numerical information present in the FP32 training data.

INT8 and INT16 can be useful when suitable scaling and calibration are
applied, but the analysis shows that a single per-tensor scale cannot
preserve the complete dynamic range of several training tensors.

Therefore, the numerical evidence supports the use of a mixed-precision
strategy rather than replacing FP32 universally. Reduced precision is most
promising for forward-path and storage-oriented quantities, whereas
gradients and especially weight updates require greater numerical care.

These conclusions are based on numerical range, observed spacing, and
precision analysis. They do not establish identical model accuracy after
precision reduction. Demonstrating accuracy equivalence would require an
end-to-end lower-precision implementation and evaluation, which was not
required by the assignment.


8. METHODOLOGICAL NOTE

The ResNet-18/CIFAR-10 training experiment was performed in FP32 for
30 epochs, with measurements collected at the beginning, middle, and end
of training.

The inference analyses for ResNet-18/CIFAR-10, MobileNetV2/CIFAR-100, and
LeNet-5/MNIST were also performed using FP32 computations.

The lower-precision formats were not used to implement or train the models.
They were evaluated only through numerical range and resolution analysis,
consistent with the assignment requirement that implementation in other
data representations was not expected.
