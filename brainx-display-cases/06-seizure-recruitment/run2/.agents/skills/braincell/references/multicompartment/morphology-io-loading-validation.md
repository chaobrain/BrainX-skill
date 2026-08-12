# BrainCell morphology IO, loading paths, and validation

Use this reference after opening `references/multicompartment/multicompartment-cell-workflow.md` when geometry must come from SWC, Neurolucida ASC, NeuroML2, NeuroMorpho.Org, or a BrainCell checkpoint. Keep manual branch construction in `references/multicompartment/braincell-manual-morphology-construction.md`.

## Choose a loading path

Each reader returns the same simulation-facing `Morphology`; choose the path from the source format and the validation evidence the task requires.

| API | Use when | Important behavior and result |
|---|---|---|
| `Morphology.from_swc(path, options=..., mode=..., return_report=False)` | The reconstruction is an SWC point/parent table. | It maps SWC structure identifiers to typed branches, validates the tree, and returns a `Morphology`; with `return_report=True`, it returns `(morphology, SwcReport)`. |
| `Morphology.from_asc(path, return_report=False)` | The reconstruction is Neurolucida ASC and its spines, markers, contours, or metadata matter. | It parses the richer nested format and returns a `Morphology`; with `return_report=True`, it also returns an `AscReport`. |
| `NeuroMlReader().read(path)` | The source is NeuroML2. | It imports segment groups and cable geometry into a `Morphology`; it does not import channel or network definitions. |
| `Morphology.from_neuromorpho(neuron_id, ...)` | A known NeuroMorpho.Org integer neuron id should be downloaded, cached, parsed, and returned directly. | It reuses the cached download on later calls and returns a `Morphology`. |
| `braincell.io.load_neuromorpho(neuron_id, return_report=False)` | The NeuroMorpho path also needs the IO-module surface or a validation report. | It returns the same morphology path; with `return_report=True`, it returns `(morphology, report)`. |
| `braincell.io.load_morpho(path)` | A processed BrainCell morphology checkpoint should be reused without parsing the original reconstruction. | It returns the stored `Morphology` or raises a checkpoint compatibility error. |

Use SWC for the portable default interchange path. Use ASC when its additional annotations must survive import. Use a BrainCell checkpoint only after the source reconstruction has already been parsed, reviewed, and optionally edited.

## Load and validate SWC

An SWC load is a parse-and-validation operation; request the report whenever malformed geometry, automatic repairs, or structure-type mapping could change the model.

| API | Description |
|---|---|
| `SwcReadOptions(standardize_safe_fixes=..., unknown_type_as_custom=..., require_root_type_soma=...)` | Use to choose whether safe corrections are applied, unknown structure ids become `CustomBranch`, and the root must be a soma. |
| `Morphology.from_swc(path, options=options, return_report=True)` | Use for the normal load-and-review path; it returns the morphology and the findings or fixes in `SwcReport`. |
| `SwcReader().check(path)` | Use when the file must be validated without constructing a `Morphology`; it returns the report. |
| `SwcReader().read(path)` | Use only when the lower-level reader is needed directly; it parses and returns the morphology. |

```python
import braincell
from braincell.filter import branch_in
from braincell.io import SwcReadOptions


options = SwcReadOptions(
    standardize_safe_fixes=True,
    unknown_type_as_custom=True,
    require_root_type_soma=True,
)
morphology, report = braincell.Morphology.from_swc(
    "neuron.swc",
    options=options,
    return_report=True,
)

print(report)
print(morphology.topo())

soma = morphology.select(branch_in("type", "soma"))
assert soma.intervals, "validated morphology has no branch typed as soma"
```

SWC ids `1`, `2`, `3`, and `4` map to soma, axon, basal dendrite, and apical dendrite. Other ids become custom branches only when the chosen reader policy permits it. Inspect the report before accepting disconnected points, a missing soma, unknown types, or applied repairs.

## Handle format-specific behavior

### Neurolucida ASC

Use the high-level constructor unless lower-level ASC models are required.

| API | Description |
|---|---|
| `Morphology.from_asc(path, return_report=True)` | Load typed branches and return the morphology with an `AscReport` containing issues and metadata. |
| `AscReader().read(path)` | Parse ASC through the lower-level reader and return the morphology. |
| `AscSpineRecord` / `AscMetadata` | Inspect spine annotations or file-level metadata when those ASC-only details affect preprocessing. |

Do not assume a successful ASC parse makes coincident points, radii, branch labels, or imported annotations appropriate for the intended simulation.

### NeuroML2

Use `NeuroMlReader` only for morphology import:

```python
from braincell.io import NeuroMlReader


morphology = NeuroMlReader().read("neuron.cell.nml")
```

Declare ions, channels, clamps, probes, and network structure separately. BrainCell does not import those NeuroML2 definitions automatically.

### NeuroMorpho.Org

Install the IO extra before using the repository client: `python -m pip install -U "braincell[io]"`.

| API | Description |
|---|---|
| `Morphology.from_neuromorpho(neuron_id, ...)` | Use the shortest path for a known integer neuron id; it downloads once, caches, parses, and returns the morphology. |
| `NeuroMorphoClient().iter_search(**criteria)` | Search by criteria such as species, brain region, or cell type; it yields typed neuron records. |
| `NeuroMorphoClient().get_neuron(neuron_id)` | Retrieve the detailed record for one integer neuron id. |
| `NeuroMorphoClient().download(neuron_id)` | Download a reconstruction and return its download record. |
| `braincell.io.fetch_neuromorpho(neuron_id, mode="standard")` | Fetch raw files without parsing; use `"standard"`, `"original"`, or `"both"` and receive a `NeuroMorphoDownloadRecord`. |
| `cache_dir=...` | Use on repository entry points when the default user cache must be replaced with an explicit location. |

Use the integer id from the neuron's NeuroMorpho.Org URL, not its display name. Treat cached content as an IO optimization, not as validation evidence.

## Save and reload processed geometry

Checkpoint a cleaned or edited BrainCell object when the exact processed geometry must be reused.

| API | Description |
|---|---|
| `braincell.io.save_morpho(morphology, path)` | Serialize a morphology and return the written `Path`. |
| `braincell.io.load_morpho(path)` | Reload a morphology checkpoint without reparsing its original source. |
| `braincell.io.save_branch(branch, path)` | Serialize one branch. |
| `braincell.io.load_branch(path)` | Reload one branch checkpoint. |
| `CheckpointError` | Catch when a checkpoint cannot be read. |
| `CheckpointVersionError` | Catch when the file format version is unsupported by the installed BrainCell. |

Do not use a checkpoint to hide the original validation record. Preserve the source identifier, reader options, report, BrainCell version, and any edits with the model provenance.

## Verify the loaded morphology

A successful load proves that the reader returned a `Morphology` under the chosen validation policy; it does not prove that the reconstruction is biologically or numerically suitable.

1. Read the validation report and distinguish detected issues from applied fixes.
2. Inspect `morphology.topo()` for the intended root, connectivity, names, and parent-child structure.
3. Select soma, axon, and dendrite types with `morphology.select(...)`; confirm that later region expressions will reach the intended branches.
4. Inspect coordinates, lengths, and radii for zero-length, disconnected, or implausible geometry.
5. Visualize the branch tree before choosing a CV policy or attaching mechanisms.
6. Record the reader options, input identity, and checkpoint version with the simulation.

Open `references/multicompartment/topology-building-and-visualization.md` when branch, CV, or runtime placement must be inspected visually. Return to `references/multicompartment/multicompartment-cell-workflow.md` after validation to choose a CV policy and declare the cell.

## Sources

- [IO overview](https://brainx.chaobrain.com/braincell/file_formats/overview.html)
- [SWC](https://brainx.chaobrain.com/braincell/file_formats/swc.html)
- [Neurolucida ASC](https://brainx.chaobrain.com/braincell/file_formats/asc.html)
- [NeuroML2](https://brainx.chaobrain.com/braincell/file_formats/neuroml2.html)
- [NeuroMorpho.Org](https://brainx.chaobrain.com/braincell/file_formats/neuromorpho.html)
- [Checkpointing](https://brainx.chaobrain.com/braincell/file_formats/checkpointing.html)
- [Morphology](https://brainx.chaobrain.com/braincell/concepts/morphology.html)
