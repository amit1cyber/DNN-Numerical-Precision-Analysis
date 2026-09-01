import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 1
BATCH_SIZE = 128
LR = 0.1

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

ROOT = "data"

os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

print("Device:", DEVICE)


# ============================================================
# DATASETS
# ============================================================

cifar10_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

cifar100_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

mnist_tf = transforms.ToTensor()


# CIFAR-10
cifar10_train = datasets.CIFAR10(
    ROOT,
    train=True,
    download=False,
    transform=cifar10_tf
)

cifar10_test = datasets.CIFAR10(
    ROOT,
    train=False,
    download=False,
    transform=cifar10_tf
)


# CIFAR-100
cifar100_test = datasets.CIFAR100(
    ROOT,
    train=False,
    download=False,
    transform=cifar100_tf
)


# MNIST
mnist_test = datasets.MNIST(
    ROOT,
    train=False,
    download=True,
    transform=mnist_tf
)


cifar10_train_loader = DataLoader(
    cifar10_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

cifar10_test_loader = DataLoader(
    cifar10_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

cifar100_test_loader = DataLoader(
    cifar100_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

mnist_test_loader = DataLoader(
    mnist_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LE NET-5
# ============================================================

class LeNet5(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 6, kernel_size=5
        )

        self.pool1 = nn.AvgPool2d(2)

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            6, 16, kernel_size=5
        )

        self.pool2 = nn.AvgPool2d(2)

        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(
            16 * 4 * 4,
            120
        )

        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(
            120,
            84
        )

        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(
            84,
            10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)

        x = self.fc3(x)

        return x


# ============================================================
# GENERIC LAYER HOOK
# ============================================================

COMPUTE_LAYERS = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear
)


class Collector:

    def __init__(self):

        self.data = {}

    def hook(self, name):

        def fn(module, inputs, output):

            x = inputs[0].detach().cpu().numpy().flatten()
            y = output.detach().cpu().numpy().flatten()

            # Keep memory under control.
            if len(x) > 30000:
                x = np.random.choice(
                    x, 30000, replace=False
                )

            if len(y) > 30000:
                y = np.random.choice(
                    y, 30000, replace=False
                )

            self.data[name] = {
                "input": x,
                "activation": y
            }

        return fn

    def register(self, model):

        for name, layer in model.named_modules():

            if isinstance(layer, COMPUTE_LAYERS):

                layer.register_forward_hook(
                    self.hook(name)
                )


# ============================================================
# NUMERICAL STATISTICS
# ============================================================

def statistics(x):

    x = np.asarray(
        x,
        dtype=np.float64
    ).flatten()

    if len(x) == 0:

        return {
            "minimum": 0,
            "maximum": 0,
            "smallest_nonzero": 0,
            "zero_percent": 0,
            "minimum_spacing": 0,
            "maximum_abs": 0
        }

    nonzero = np.abs(
        x[x != 0]
    )

    unique = np.unique(x)

    if len(unique) > 1:

        spacing = np.diff(unique)

        spacing = spacing[
            spacing > 0
        ]

        min_spacing = (
            np.min(spacing)
            if len(spacing)
            else 0
        )

    else:

        min_spacing = 0

    return {
        "minimum": np.min(x),
        "maximum": np.max(x),

        "smallest_nonzero":
            np.min(nonzero)
            if len(nonzero)
            else 0,

        "zero_percent":
            100 * np.mean(x == 0),

        "minimum_spacing":
            min_spacing,

        "maximum_abs":
            np.max(np.abs(x))
    }


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(data, filename):

    rows = []

    for name, values in data.items():

        s = statistics(values)

        rows.append({
            "layer": name,
            **s
        })

    pd.DataFrame(rows).to_csv(
        f"results/csv/{filename}.csv",
        index=False
    )


# ============================================================
# PLOT DISTRIBUTIONS
# ============================================================

def plot_distributions(data, prefix):

    for name, values in data.items():

        values = np.asarray(
            values,
            dtype=np.float32
        ).flatten()

        if len(values) == 0:
            continue

        # Sample only for plotting
        if len(values) > 200000:

            values = np.random.choice(
                values,
                200000,
                replace=False
            )

        safe = (
            name.replace(".", "_")
                .replace("/", "_")
        )

        # ----------------------------------------------------
        # FULL RANGE
        # ----------------------------------------------------

        plt.figure(figsize=(7, 4))

        plt.hist(
            values,
            bins=200,
            density=True
        )

        plt.xlabel("Numerical value")
        plt.ylabel("Probability density")
        plt.title(
            f"{name} - Full numerical range"
        )

        plt.tight_layout()

        plt.savefig(
            f"results/plots/"
            f"{prefix}_{safe}_full.png",
            dpi=120
        )

        plt.close()

        # ----------------------------------------------------
        # AROUND ZERO
        # ----------------------------------------------------

        r = np.percentile(
            np.abs(values),
            99
        )

        if r > 0:

            plt.figure(figsize=(7, 4))

            plt.hist(
                values,
                bins=200,
                density=True
            )

            plt.xlim(
                -r,
                r
            )

            plt.xlabel("Numerical value")
            plt.ylabel("Probability density")
            plt.title(
                f"{name} - Magnified around zero"
            )

            plt.tight_layout()

            plt.savefig(
                f"results/plots/"
                f"{prefix}_{safe}_zero.png",
                dpi=120
            )

            plt.close()


# ============================================================
# TRAINING ANALYSIS
# ============================================================

print("\n======================================")
print("TRAINING: RESNET-18 / CIFAR-10")
print("======================================")

resnet = models.resnet18(
    num_classes=10
).float().to(DEVICE)

collector = Collector()
collector.register(resnet)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    resnet.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=5e-4
)

# Beginning / middle / end
STAGES = {
    1: "beginning",
    EPOCHS // 2: "middle",
    EPOCHS: "end"
}


for epoch in range(
    1,
    EPOCHS + 1
):

    resnet.train()

    collect = (
        epoch in STAGES
    )

    if collect:

        stage = STAGES[epoch]

        inputs = {}
        activations = {}
        gradients = {}
        updates = {}
        weights = {}
        biases = {}

    for images, labels in cifar10_train_loader:

        images = images.to(
            DEVICE,
            dtype=torch.float32
        )

        labels = labels.to(DEVICE)

        # Save weights before update
        old_weights = {}

        if collect:

            for name, p in resnet.named_parameters():

                old_weights[name] = (
                    p.detach().clone()
                )

        optimizer.zero_grad()

        output = resnet(images)

        loss = criterion(
            output,
            labels
        )

        loss.backward()

        # ----------------------------------------------------
        # GRADIENTS
        # ----------------------------------------------------

        if collect:

            for name, p in resnet.named_parameters():

                if p.grad is not None:

                    gradients[name] = (
                        p.grad
                         .detach()
                         .cpu()
                         .numpy()
                         .flatten()
                    )

        optimizer.step()

        # ----------------------------------------------------
        # WEIGHT UPDATES
        # ----------------------------------------------------

        if collect:

            for name, p in resnet.named_parameters():

                updates[name] = (
                    p.detach()
                     .cpu()
                     .numpy()
                     .flatten()
                    -
                    old_weights[name]
                     .cpu()
                     .numpy()
                     .flatten()
                )

    # ========================================================
    # COLLECT STAGE DATA
    # ========================================================

    if collect:

        resnet.eval()

        # Parameters
        for name, p in resnet.named_parameters():

            v = (
                p.detach()
                 .cpu()
                 .numpy()
                 .flatten()
            )

            if "bias" in name:
                biases[name] = v
            else:
                weights[name] = v

        # Run one batch to capture inputs/activations
        collector.data.clear()

        images, labels = next(
            iter(cifar10_train_loader)
        )

        images = images.to(
            DEVICE,
            dtype=torch.float32
        )

        with torch.no_grad():

            resnet(images)

        # Every layer's input
        for name, d in collector.data.items():

            inputs[name] = d["input"]

            activations[name] = (
                d["activation"]
            )

        # Save
        save_statistics(
            inputs,
            f"training_{stage}_inputs"
        )

        save_statistics(
            weights,
            f"training_{stage}_weights"
        )

        save_statistics(
            activations,
            f"training_{stage}_activations"
        )

        save_statistics(
            gradients,
            f"training_{stage}_gradients"
        )

        save_statistics(
            updates,
            f"training_{stage}_updates"
        )

        save_statistics(
            biases,
            f"training_{stage}_biases"
        )

        # Plots
        plot_distributions(
            inputs,
            f"training_{stage}_inputs"
        )

        plot_distributions(
            weights,
            f"training_{stage}_weights"
        )

        plot_distributions(
            activations,
            f"training_{stage}_activations"
        )

        plot_distributions(
            gradients,
            f"training_{stage}_gradients"
        )

        plot_distributions(
            updates,
            f"training_{stage}_updates"
        )

        plot_distributions(
            biases,
            f"training_{stage}_biases"
        )

        print(
            f"Collected: {stage}"
        )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"Loss={loss.item():.4f}"
    )


# ============================================================
# GENERIC INFERENCE FUNCTION
# ============================================================

def run_inference(
    model,
    loader,
    model_name
):

    print("\n======================================")
    print(
        f"INFERENCE: {model_name}"
    )
    print("======================================")

    model = model.float().to(DEVICE)
    model.eval()

    collector = Collector()
    collector.register(model)

    # --------------------------------------------------------
    # Weights / biases
    # --------------------------------------------------------

    weights = {}
    biases = {}

    for name, p in model.named_parameters():

        values = (
            p.detach()
             .cpu()
             .numpy()
             .flatten()
        )

        if "bias" in name:
            biases[name] = values
        else:
            weights[name] = values

    # --------------------------------------------------------
    # Inputs / activations / logits
    # --------------------------------------------------------

    inputs = []
    activation_store = {}
    logits = []

    with torch.no_grad():

        for batch_no, (
            images,
            labels
        ) in enumerate(loader):

            images = images.to(
                DEVICE,
                dtype=torch.float32
            )

            collector.data.clear()

            output = model(images)

            # Input
            x = (
                images.cpu()
                      .numpy()
                      .flatten()
            )

            if len(x) > 30000:

                x = np.random.choice(
                    x,
                    30000,
                    replace=False
                )

            inputs.append(x)

            # Activations
            for name, d in collector.data.items():

                if name not in activation_store:

                    activation_store[name] = []

                v = d["activation"]

                activation_store[
                    name
                ].append(v)

            # Logits
            logits.append(
                output.cpu()
                      .numpy()
                      .flatten()
            )

            if batch_no % 20 == 0:

                print(
                    f"Processed "
                    f"{batch_no + 1}/"
                    f"{len(loader)} batches"
                )

    inputs = {
        "network_input":
            np.concatenate(inputs)
    }

    activations = {}

    for name, values in activation_store.items():

        v = np.concatenate(values)

        # Limit final memory footprint
        if len(v) > 500000:

            v = np.random.choice(
                v,
                500000,
                replace=False
            )

        activations[name] = v

    logits = {
        "output_logits":
            np.concatenate(logits)
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    prefix = model_name.replace(
        " ",
        "_"
    ).lower()

    save_statistics(
        inputs,
        f"{prefix}_inputs"
    )

    save_statistics(
        weights,
        f"{prefix}_weights"
    )

    save_statistics(
        activations,
        f"{prefix}_activations"
    )

    save_statistics(
        biases,
        f"{prefix}_biases"
    )

    save_statistics(
        logits,
        f"{prefix}_logits"
    )

    plot_distributions(
        inputs,
        f"{prefix}_inputs"
    )

    plot_distributions(
        weights,
        f"{prefix}_weights"
    )

    plot_distributions(
        activations,
        f"{prefix}_activations"
    )

    plot_distributions(
        biases,
        f"{prefix}_biases"
    )

    plot_distributions(
        logits,
        f"{prefix}_logits"
    )

    return (
        inputs,
        weights,
        activations,
        biases,
        logits
    )


# ============================================================
# RESNET-18 INFERENCE
# ============================================================

run_inference(
    resnet,
    cifar10_test_loader,
    "ResNet18_CIFAR10"
)


# ============================================================
# MOBILENETV2 / CIFAR-100
# ============================================================

mobilenet = models.mobilenet_v2(
    num_classes=100
).float()

run_inference(
    mobilenet,
    cifar100_test_loader,
    "MobileNetV2_CIFAR100"
)


# ============================================================
# LENET-5 / MNIST
# ============================================================

lenet = LeNet5().float()

run_inference(
    lenet,
    mnist_test_loader,
    "LeNet5_MNIST"
)


# ============================================================
# LOWER PRECISION ANALYSIS
# ============================================================

def lower_precision_analysis(
    x,
    name
):

    x = np.asarray(
        x,
        dtype=np.float32
    ).flatten()

    if len(x) > 200000:

        x = np.random.choice(
            x,
            200000,
            replace=False
        )

    result = {
        "data": name
    }

    # ========================================================
    # FP16
    # ========================================================

    fp16 = (
        x.astype(np.float16)
         .astype(np.float32)
    )

    result["FP16_MAE"] = np.mean(
        np.abs(x - fp16)
    )

    result["FP16_max_error"] = np.max(
        np.abs(x - fp16)
    )

    # ========================================================
    # INT8
    # ========================================================

    max_value = np.max(
        np.abs(x)
    )

    if max_value > 0:

        scale8 = max_value / 127

        q8 = np.round(
            x / scale8
        )

        q8 = np.clip(
            q8,
            -127,
            127
        )

        reconstructed8 = (
            q8 * scale8
        )

        result["INT8_MAE"] = np.mean(
            np.abs(
                x - reconstructed8
            )
        )

        result["INT8_max_error"] = np.max(
            np.abs(
                x - reconstructed8
            )
        )

        result["INT8_zero_percent"] = (
            100 *
            np.mean(q8 == 0)
        )

        # ====================================================
        # INT16
        # ====================================================

        scale16 = (
            max_value / 32767
        )

        q16 = np.round(
            x / scale16
        )

        q16 = np.clip(
            q16,
            -32767,
            32767
        )

        reconstructed16 = (
            q16 * scale16
        )

        result["INT16_MAE"] = np.mean(
            np.abs(
                x - reconstructed16
            )
        )

        result["INT16_max_error"] = np.max(
            np.abs(
                x - reconstructed16
            )
        )

        result["INT16_zero_percent"] = (
            100 *
            np.mean(q16 == 0)
        )

    return result


# ============================================================
# COMBINE PRECISION RESULTS
# ============================================================

precision_rows = []

# ============================================================
# RESNET-18 INFERENCE
# ============================================================

resnet_data = run_inference(
    resnet,
    cifar10_test_loader,
    "ResNet18_CIFAR10"
)


# ============================================================
# MOBILENETV2 / CIFAR-100
# ============================================================

mobilenet = models.mobilenet_v2(
    num_classes=100
).float()

mobilenet_data = run_inference(
    mobilenet,
    cifar100_test_loader,
    "MobileNetV2_CIFAR100"
)


# ============================================================
# LENET-5 / MNIST
# ============================================================

lenet = LeNet5().float()

lenet_data = run_inference(
    lenet,
    mnist_test_loader,
    "LeNet5_MNIST"
)


# ============================================================
# LOWER-PRECISION ANALYSIS
# ============================================================

def lower_precision_analysis(x, name):

    x = np.asarray(
        x,
        dtype=np.float32
    ).flatten()

    if len(x) > 200000:

        x = np.random.choice(
            x,
            200000,
            replace=False
        )

    result = {
        "data": name
    }

    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    fp16 = (
        x.astype(np.float16)
         .astype(np.float32)
    )

    result["FP16_MAE"] = np.mean(
        np.abs(x - fp16)
    )

    result["FP16_max_error"] = np.max(
        np.abs(x - fp16)
    )

    # --------------------------------------------------------
    # INT8
    # --------------------------------------------------------

    max_value = np.max(np.abs(x))

    if max_value > 0:

        scale8 = max_value / 127

        q8 = np.round(
            x / scale8
        )

        q8 = np.clip(
            q8,
            -127,
            127
        )

        reconstructed8 = (
            q8 * scale8
        )

        result["INT8_MAE"] = np.mean(
            np.abs(
                x - reconstructed8
            )
        )

        result["INT8_max_error"] = np.max(
            np.abs(
                x - reconstructed8
            )
        )

        result["INT8_zero_percent"] = (
            100 * np.mean(q8 == 0)
        )

        # ----------------------------------------------------
        # INT16
        # ----------------------------------------------------

        scale16 = max_value / 32767

        q16 = np.round(
            x / scale16
        )

        q16 = np.clip(
            q16,
            -32767,
            32767
        )

        reconstructed16 = (
            q16 * scale16
        )

        result["INT16_MAE"] = np.mean(
            np.abs(
                x - reconstructed16
            )
        )

        result["INT16_max_error"] = np.max(
            np.abs(
                x - reconstructed16
            )
        )

        result["INT16_zero_percent"] = (
            100 * np.mean(q16 == 0)
        )

    # --------------------------------------------------------
    # FP8 RANGE ANALYSIS
    # --------------------------------------------------------
    #
    # FP8 is analyzed theoretically/post-hoc.
    # No FP8 model computation is performed.
    #
    # E4M3 approximate max = 448
    # E5M2 approximate max = 57344
    #
    # Minimum normal values:
    # E4M3 ~= 2^-6
    # E5M2 ~= 2^-14
    #
    # Subnormal values extend the usable range further.
    # --------------------------------------------------------

    max_abs = np.max(np.abs(x))

    result["FP8_E4M3_range_ok"] = (
        max_abs <= 448
    )

    result["FP8_E5M2_range_ok"] = (
        max_abs <= 57344
    )

    return result


# ============================================================
# RUN PRECISION ANALYSIS
# ============================================================

all_precision_rows = []


all_models = {
    "ResNet18_CIFAR10": resnet_data,
    "MobileNetV2_CIFAR100": mobilenet_data,
    "LeNet5_MNIST": lenet_data
}


for model_name, data in all_models.items():

    inputs, weights, activations, biases, logits = data

    for dtype, collection in [

        ("inputs", inputs),
        ("weights", weights),
        ("activations", activations),
        ("biases", biases),
        ("logits", logits)

    ]:

        for layer_name, values in collection.items():

            result = lower_precision_analysis(
                values,
                f"{model_name}/{dtype}/{layer_name}"
            )

            all_precision_rows.append(
                result
            )


precision_df = pd.DataFrame(
    all_precision_rows
)

precision_df.to_csv(
    "results/csv/"
    "lower_precision_analysis.csv",
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==============================================")
print("ALL NUMERICAL ANALYSIS COMPLETE")
print("==============================================")
print()
print("Training:")
print("  ResNet-18 / CIFAR-10")
print("  Beginning / Middle / End")
print()
print("Inference:")
print("  ResNet-18 / CIFAR-10")
print("  MobileNetV2 / CIFAR-100")
print("  LeNet-5 / MNIST")
print()
print("Results:")
print("  results/csv/")
print("  results/plots/")
print()
print("All neural-network computation was performed")
print("using FP32.")
print("==============================================")