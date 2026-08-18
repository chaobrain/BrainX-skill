# BrainMass datasets

Use this reference when loading bundled BrainMass examples, inspecting typed connectome or signal containers, generating delayed-match task data, or registering a small local dataset loader.

## Registry

| API | Description |
|---|---|
| `brainmass.datasets.list_datasets()` | List names and descriptions currently registered in the installed release. |
| `brainmass.datasets.load_dataset(name)` | Resolve a registered loader and return its typed or generated value; an unknown name raises instead of guessing a path. |
| `brainmass.datasets.register_dataset(name, loader, description=...)` | Register a callable under one name for the current process; the callable runs each time the dataset is loaded. |

```python
import brainmass

available = brainmass.datasets.list_datasets()
connectome = brainmass.datasets.load_dataset("example_connectome")
signal = brainmass.datasets.load_dataset("example_signal")

assert connectome.weights.shape == connectome.distances.shape == (8, 8)
assert signal.signal.shape == (500, 8)
```

Use the registry for small, deterministic tutorial or study-local loaders. It is not a remote dataset manager or a persistence format.

## Typed containers

| API | Description |
|---|---|
| `brainmass.datasets.Connectome(weights, distances, labels)` | Hold square structural weights, unit-aware inter-region distances, and region labels used by `Network`. |
| `brainmass.datasets.Signal(signal, dt, labels, fc)` | Hold a time-major multi-region signal, its sampling step, labels, and precomputed functional connectivity. |
| `brainmass.datasets.delayed_match_task(n_samples=..., seq_len=..., n_symbols=..., seed=...)` | Generate reproducible delayed-match-to-sample inputs and binary targets for the HORN workflow. |

The built-in registry provides:

| Name | Result | Use |
|---|---|---|
| `example_connectome` | `Connectome` | Small symmetric eight-region weights, millimeter distances, and labels for network tutorials. |
| `example_signal` | `Signal` | Short eight-region target trajectory, sampling `dt`, labels, and FC for fitting and analysis. |
| `delayed_match_task` | `(inputs, targets)` | Synthetic task data generated from the requested size and seed. |

```python
inputs, targets = brainmass.datasets.delayed_match_task(
    n_samples=32,
    seq_len=8,
    n_symbols=2,
    seed=0,
)

assert inputs.shape == (32, 8, 2)
assert targets.shape == (32,)
```

Treat BrainMass signals as time-major and HORN task inputs as `(sample, time, feature)` until the training workflow explicitly transposes them.

## Register a loader

```python
brainmass.datasets.register_dataset(
    "study_connectome",
    lambda: connectome,
    description="Validated connectome for the current study",
)
loaded = brainmass.datasets.load_dataset("study_connectome")
assert loaded is connectome
```

Validate shape, symmetry or directionality, diagonal policy, units, labels, and sampling interval in the loader or immediately after loading. Registration alone does not validate scientific metadata.

## Official source

- `https://brainx.chaobrain.com/brainmass/reference/datasets.html`
