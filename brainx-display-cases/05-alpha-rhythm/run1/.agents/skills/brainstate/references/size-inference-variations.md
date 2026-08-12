# Size inference and composition

Use this reference to compose size-aware layers with `Sequential` and `.desc()`, or when stride, padding, dimensionality, pooling mode, or flatten axes make a layer's size behavior non-obvious.

## Selection map

| Need | Use | Key constraint |
|---|---|---|
| Infer each next layer from the previous output | Give the first layer `in_size`, then use `.desc()` | Every descriptor must receive enough non-size constructor arguments to instantiate later. |
| Predict convolution output | Inspect `stride`, `padding`, dilation, rank, and channel layout | Use `stride`, not the obsolete `strides` spelling. |
| Reduce by local windows | `AvgPool*`, `MaxPool*`, or `LPPool*` | Kernel, stride, padding, and channel axis determine the result. |
| Produce a fixed spatial target | `AdaptiveAvgPool*` or `AdaptiveMaxPool*` | `target_size` determines the spatial output independently of input size. |
| Flatten a runtime batch | Set `start_axis` to preserve the batch axis | Runtime axes include batch; Module `in_size` metadata does not. |
| Flatten feature metadata for composition | Construct `Flatten(in_size=feature_size)` or use `.desc()` | Feature `in_size` excludes the batch dimension. |

## Compose `ComplexNet` with `Sequential` and `.desc()`

Give the first layer an explicit `in_size`, then use `.desc()` wherever `Sequential` should instantiate a layer from the preceding `out_size`. Start the classifier from `self.features.out_size` so flattening and dense construction stay coupled to the feature extractor.

```python
import brainstate
import brainstate.nn as nn


class ComplexNet(brainstate.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_size,
                out_channels=16,
                kernel_size=3,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.Conv2d.desc(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.Conv2d.desc(
                out_channels=64,
                kernel_size=3,
                padding="SAME",
            ),
            nn.ReLU(),
            nn.MaxPool2d.desc(
                kernel_size=(2, 2),
                stride=(2, 2),
                channel_axis=-1,
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(in_size=self.features.out_size),
            nn.Linear.desc(out_size=256),
            nn.ReLU(),
            nn.Linear.desc(out_size=10),
        )

    def update(self, x):
        return self.classifier(self.features(x))


brainstate.random.seed(42)
net = ComplexNet(in_size=(32, 32, 3))
x_image = brainstate.random.randn(2, 32, 32, 3)
y_image = net(x_image)

assert net.features.out_size == (8, 8, 64)
assert net.classifier.layers[0].out_size == (4096,)
assert net.classifier.out_size == (10,)
assert y_image.shape == (2, 10)
```

## Convolution output-size behavior

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

The tutorial's **Convolution Parameters** rules are:

- `stride` controls the step size; pass it with the `stride` keyword. The API no longer accepts `strides`.
- `'SAME'` preserves spatial dimensions when stride is 1; `'VALID'` performs no padding.
- Explicit tuples set different values per spatial dimension.

The following source pattern bundles those rules with the exact demonstrated 28-by-28, 3-by-3-kernel outcomes:

```python
import brainstate

x = brainstate.random.randn(1, 28, 28, 3)

configs = [
    {"stride": (1, 1), "padding": "SAME", "name": "Stride 1, SAME"},
    {"stride": (2, 2), "padding": "SAME", "name": "Stride 2, SAME"},
    {"stride": (1, 1), "padding": "VALID", "name": "Stride 1, VALID"},
]

for config in configs:
    brainstate.random.seed(0)
    conv = brainstate.nn.Conv2d(
        in_size=(28, 28, 3),
        out_channels=16,
        kernel_size=(3, 3),
        stride=config["stride"],
        padding=config["padding"],
    )
    y = conv(x)
    print(f"{config['name']:20s}: out_size={conv.out_size}, runtime={y.shape}")
```

Expected output:

```text
Stride 1, SAME      : out_size=(28, 28, 16), runtime=(1, 28, 28, 16)
Stride 2, SAME      : out_size=(14, 14, 16), runtime=(1, 14, 14, 16)
Stride 1, VALID     : out_size=(26, 26, 16), runtime=(1, 26, 26, 16)
```

These are the tutorial's exact cases: `'SAME'` with stride 2 produces 14 by 14 from this 28-by-28 input, while `'VALID'` with stride 1 and a 3-by-3 kernel produces 26 by 26. Use `stride`, not `strides`.

### Dimensionality and layout variations

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

The tutorial demonstrates channels-last layouts for each convolution rank:

| Layer | Runtime input layout | Demonstrated input | Demonstrated output |
|---|---|---|---|
| `Conv1d` | `(batch, length, channels)` | `(4, 100, 3)` | `(4, 100, 16)` |
| `Conv2d` | `(batch, height, width, channels)` | `(8, 28, 28, 3)` | `(8, 28, 28, 32)` |
| `Conv3d` | `(batch, depth, height, width, channels)` | `(2, 16, 64, 64, 3)` | `(2, 16, 64, 64, 16)` |

Each case uses `'SAME'` padding and the default stride of 1:

```python
import brainstate

brainstate.random.seed(0)

conv1d = brainstate.nn.Conv1d(
    in_size=(100, 3),
    out_channels=16,
    kernel_size=3,
    padding="SAME",
)
conv2d = brainstate.nn.Conv2d(
    in_size=(28, 28, 3),
    out_channels=32,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="SAME",
)
conv3d = brainstate.nn.Conv3d(
    in_size=(16, 64, 64, 3),
    out_channels=16,
    kernel_size=(3, 3, 3),
    padding="SAME",
)

print(conv1d(brainstate.random.randn(4, 100, 3)).shape)
print(conv2d(brainstate.random.randn(8, 28, 28, 3)).shape)
print(conv3d(brainstate.random.randn(2, 16, 64, 64, 3)).shape)
```

## Pooling reductions and fixed targets

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

Official source phrase: "Pooling layers downsample feature maps, reducing spatial dimensions." `MaxPool2d` takes the maximum value in each window; `AvgPool2d` takes the average. With a 2-by-2 kernel and stride 2, both demonstrated layers reduce 28 by 28 to 14 by 14 and retain all 16 channels:

```python
import brainstate

x = brainstate.random.randn(4, 28, 28, 16)
maxpool = brainstate.nn.MaxPool2d(
    kernel_size=(2, 2),
    stride=(2, 2),
)
avgpool = brainstate.nn.AvgPool2d(
    kernel_size=(2, 2),
    stride=(2, 2),
)

print(maxpool(x).shape)  # (4, 14, 14, 16)
print(avgpool(x).shape)  # (4, 14, 14, 16)
```

The printed layer representations in the tutorial show `padding=VALID` and `channel_axis=-1` for both constructors.

Use `AdaptiveAvgPool2d` when the required spatial result is fixed rather than tied to one input resolution. The tutorial's `target_size=(7, 7)` example maps all three inputs to 7 by 7:

```python
adaptive_pool = brainstate.nn.AdaptiveAvgPool2d(target_size=(7, 7))

inputs = [
    brainstate.random.randn(1, 28, 28, 16),
    brainstate.random.randn(1, 56, 56, 16),
    brainstate.random.randn(1, 224, 224, 16),
]

for x in inputs:
    y = adaptive_pool(x)
    print(f"Input {x.shape[1:3]} -> Output {y.shape[1:3]}")
```

Expected output:

```text
Input (28, 28) -> Output (7, 7)
Input (56, 56) -> Output (7, 7)
Input (224, 224) -> Output (7, 7)
```

### Pooling with constructor-visible size metadata

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

The standalone pooling examples omit `in_size` and demonstrate reduction through runtime shapes. In the tutorial's complete CNN, the final pool instead receives the preceding convolution's size so the following `Flatten` can be constructed from its result:

```python
conv3 = brainstate.nn.Conv2d(
    (8, 8, 64),
    out_channels=128,
    kernel_size=(3, 3),
    padding="SAME",
)
pool3 = brainstate.nn.MaxPool2d(
    kernel_size=(2, 2),
    stride=(2, 2),
    in_size=conv3.out_size,
)

print(conv3.out_size)  # (8, 8, 128)
print(pool3.out_size)  # (4, 4, 128)
```

## Flatten: runtime axes versus constructor inference

Source URL: https://brainx.chaobrain.com/brainstate/tutorials/core/03_common_layers.html

The standalone tutorial pattern flattens a batched tensor from `start_axis=1`, preserving its leading batch axis:

```python
import brainstate

flatten_runtime = brainstate.nn.Flatten(start_axis=1)
x_conv = brainstate.random.randn(4, 7, 7, 64)
x_flat = flatten_runtime(x_conv)

print(x_conv.shape)  # (4, 7, 7, 64)
print(x_flat.shape)  # (4, 3136)
```

The exact flattened feature count is `7 * 7 * 64 = 3136` per sample. The tutorial summary identifies `start_axis` and `end_axis` as the key `Flatten` parameters.

For constructor-aware composition, the complete CNN passes the final pool's result to `Flatten`. Its printed representation shows `start_axis=0`, `end_axis=-1`, and `out_size=(2048,)` for the feature shape `(4, 4, 128)`. Continue from the `conv3` / `pool3` construction above:

```python
import jax.numpy as jnp

flatten = brainstate.nn.Flatten(in_size=pool3.out_size)
fc1 = brainstate.nn.Linear(4 * 4 * 128, (256,))

x = brainstate.random.randn(2, 8, 8, 64)
x = conv3(x)
x = jnp.maximum(0, x)
x = pool3(x)
x = flatten(x)
y = fc1(x)

print(flatten.in_size)   # (4, 4, 128)
print(flatten.out_size)  # (2048,)
print(x.shape)           # (2, 2048)
print(y.shape)           # (2, 256)
```

This is the source's manual-composition variation: spatial feature shapes progress from `(8, 8, 128)` through `(4, 4, 128)` to `(2048,)`, and the first dense layer accepts 2048 features. Use it when explicit construction is clearer than descriptor-based composition.
