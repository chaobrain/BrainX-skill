# BrainX Documentation Style and Development Guide

> Status: canonical authoring standard for new BrainX package documentation.
>
> Scope: visual style, page-level presentation, reusable components, executable examples,
> images, API rendering, local development, validation, and deployment. This guide does
> **not** prescribe the information architecture of a package. Each package may choose the
> pages and navigation groups its users need.

## 1. Source Basis and Canonical Direction

This guide generalizes the documentation source in the following repositories, inspected
on 2026-08-09:

- [BrainX](https://github.com/chaobrain/brainx/tree/main/docs)
  (`3098bf7eab251a30bb799e82983973987b4a5622`)
- [brainstate](https://github.com/chaobrain/brainstate/tree/main/docs)
  (`99bd1ac9611d081e0af4365d3ef0a4a7760cabec`)
- [braincell](https://github.com/chaobrain/braincell/tree/main/docs)
  (`c803e77142d66aa1f03e1d91eb0a7ad407501187`)
- [BrainTrace](https://github.com/chaobrain/braintrace)
  (`1b361b9c1d60139197bc6c83ada1702bb2674dab`)
- [brainpy.state](https://github.com/chaobrain/brainpy.state/tree/main/docs)
  (`92878e4c542c4665ee3db32c94dc4866c4badcb6`)
- [brainmass](https://github.com/chaobrain/brainmass/tree/main/docs)
  (`293eaeb8948474c645208b68c531b7cb5482d4fb`)

The repositories share a clear base: Sphinx, `sphinx_book_theme`, MyST-NB,
`sphinx-design`, generated API pages, and the shared BrainX header/footer extension.
They also contain historical inconsistencies. This guide resolves those inconsistencies
into one default rather than reproducing every variation.

The canonical direction is:

- use the shared BrainX extension for brand chrome;
- use `sphinx_book_theme` for the documentation surface;
- use RST for landing pages, hub pages, and generated API indexes;
- use MyST Markdown or Jupyter notebooks for narrative and executable pages;
- use MyST-NB as the only notebook renderer;
- store real notebook outputs, but do not execute notebooks during the Sphinx build;
- use NumPy-style docstrings as the default API documentation source;
- prefer standard `sphinx-design` components over package-specific HTML and CSS.

## 2. The Visual Contract

### 2.1 Overall character

BrainX documentation should feel like a scientific workbench: calm, precise, readable,
and visibly connected to the wider ecosystem. It should not look like a marketing landing
page or a collection of decorative cards.

The experience has two layers:

1. **Shared ecosystem chrome.** The BrainX header and footer establish the brand, global
   navigation, package links, responsive menu, and ecosystem identity.
2. **Documentation surface.** `sphinx_book_theme` supplies the left navigation, central
   article, right on-page table of contents, code blocks, API signatures, search, and
   responsive reading behavior.

Do not duplicate the BrainX header, footer, logo row, or ecosystem navigation inside a
package theme. The `brainx_sphinx_header` extension is the single source of truth.

### 2.2 Brand tokens

The shared header/footer currently uses these fallback tokens:

| Token | Value | Role |
| --- | --- | --- |
| Paper | `#fbf6e9` | light branded surface |
| Paper deep | `#f6efde` | header/footer background |
| Paper soft | `#f1ddd2` | subtle hover or highlight |
| Ink | `#1c1610` | primary text |
| Ink soft | `#3a2e1f` | secondary text |
| Bronze | `#5a4a2a` | secondary accent |
| Coral | `#bd5430` | primary brand accent |
| Rule | `#d8cfb8` | borders and separators |
| Rule strong | `#b9ad8e` | stronger dividers |

The shared type families are:

- display: `Source Serif 4`, then Georgia/serif;
- body and navigation: `Inter`, then system sans-serif;
- code and badges: `JetBrains Mono`, then system monospace.

These values describe the shared brand shell. Do not paste them into a package-local CSS
file to recolor the entire theme. If the documentation surface needs an ecosystem-wide
theme change, make it in a shared stylesheet and roll it out to every package together.

### 2.3 Header and footer behavior

The shared extension provides these behaviors automatically:

- sticky header with the BrainX mark and ecosystem navigation;
- split-target dropdowns: the label navigates and the caret expands;
- visible keyboard focus and accessible `aria-expanded` state;
- responsive navigation that collapses below `800px`;
- a `48px` mobile dropdown control target;
- reduced-motion handling;
- server-side insertion into built HTML to prevent first-paint flashing;
- a JavaScript fallback if server-side insertion is unavailable;
- the BrainX footer after the native Sphinx footer;
- delegation from the BrainX mobile menu button to the Sphinx sidebar control.

Do not override `.bx-*` selectors in a package repository. Do not override the theme's
native header to make it visible again. Fix shared header behavior in the extension or its
canonical assets.

## 3. Required Documentation Stack

### 3.1 File roles

Use each source format for what it renders best:

| Format | Use it for | Do not use it for |
| --- | --- | --- |
| `.rst` | landing pages, navigation hubs, API indexes, autosummary pages, compact static guides | long runnable tutorials |
| `.ipynb` | tutorials, how-to pages, scientific examples, pages whose outputs are evidence | pure navigation or generated API lists |
| `.md` | short prose guides, migration notes, contribution notes, pages that benefit from MyST roles but do not need execution | pages that pretend pasted output is executable evidence |

RST and MyST are both valid. A single page should not use raw HTML merely to imitate a
component that Sphinx or `sphinx-design` already provides.

### 3.2 `requirements-doc.txt`

Use a dedicated documentation dependency file. The following is the default baseline;
add package-specific plotting or optional-backend dependencies only when a published page
actually imports them.

```text
# Include the one dependency set that the repository actually owns, if needed.
# -r requirements.txt
# -r requirements-dev.txt

pandoc
Jinja2
sphinx>=9.0.4
myst-nb
sphinx-book-theme>=1.2.0
sphinx-copybutton>=0.5.2
sphinx-design
sphinx-math-dollar
matplotlib
brainx-sphinx-header>=0.5.0
```

Optional dependencies:

- add `sphinx-autodoc-typehints` only if the package deliberately uses the
  type-hint-first API profile described in Section 10.3;
- add `sphinx-thebe` only if the site exposes a tested live execution control;
- add `jupyter-sphinx` only if a page uses its directives;
- add `sphinx.ext.doctest` in `conf.py` when API examples are executable doctests.

Do not enable both `myst_nb` and `nbsphinx`. MyST-NB is the BrainX default notebook
renderer.

### 3.3 Canonical `docs/conf.py`

Replace the bracketed values. Keep package-specific exceptions below this common block and
comment why each exception exists.

```python
from importlib.metadata import version
from pathlib import Path
import sys

DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
sys.path.insert(0, str(REPO_DIR))

project = "[display package name]"
author = "BrainX Developers"
copyright = "[year range], BrainX Ecosystem"
release = version("[distribution-name]")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_math_dollar",
    "brainx_sphinx_header",
]

source_suffix = [".rst", ".md", ".ipynb"]
master_doc = "index"
templates_path = ["_templates"]
html_static_path = ["_static"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"

# Canonical profile: NumPy docstrings own parameter and return types.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

numfig = True
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "jupyter_execute/**",
]

html_theme = "sphinx_book_theme"
html_title = "[display package name]"
html_logo = "https://brainx.chaobrain.com/images/[package-slug].webp"
html_favicon = html_logo
html_baseurl = "https://brainx.chaobrain.com/[package-slug]/"
html_copy_source = True
html_sourcelink_suffix = ""
html_last_updated_fmt = ""

html_theme_options = {
    "show_toc_level": 2,
}

# Notebooks are executed and reviewed during authoring. Sphinx renders their
# committed outputs and does not rerun scientific workloads during the build.
nb_execution_mode = "off"
nb_execution_timeout = 200
```

Rules for this configuration:

- `html_baseurl` must be the production URL and must end in `/`.
- Keep `brainx_inject_base = True`, the extension default, unless a tested host requires
  historical redirect behavior.
- Use a package-specific `html_logo` and favicon, but keep the BrainX header brand mark
  unchanged.
- Do not add `html_css_files` for a file that is not committed.
- Do not suppress broad warning classes to make a build green. Fix warnings or add a narrow,
  documented ignore for an external type the package does not own.
- Do not delete build directories from `conf.py`. Clean them explicitly in development or CI.
- Do not copy large content trees in `conf.py`; the configuration phase should remain fast and
  predictable.

### 3.4 Generated assets and Git ignores

The shared header extension writes its cache and generated assets during the build. Ignore
them rather than committing them:

```gitignore
docs/_build/
docs/_brand/
docs/jupyter_execute/
.ipynb_checkpoints/
```

Keep authored files such as `docs/_static/...`, notebook outputs, and RST autosummary
templates under version control.

## 4. Page Layout and Typography

### 4.1 Default page anatomy

The standard desktop page has:

- the shared sticky BrainX header at the top;
- a left package navigation sidebar generated from hidden toctrees;
- one central article column;
- a right in-page table of contents showing through heading level 2;
- the native Sphinx footer followed immediately by the shared BrainX footer.

Authors control the article, not the shell. Do not add another page-level header, floating
sidebar, footer, search box, or breadcrumb implementation inside page content.

### 4.2 Headings

Use one page title and a shallow hierarchy.

RST:

```rst
Page Title
==========

Major Section
-------------

Subsection
^^^^^^^^^^
```

MyST or notebook Markdown:

```markdown
# Page Title

## Major section

### Subsection
```

Heading rules:

- use sentence case, except for proper names;
- keep the page title literal and descriptive;
- do not begin a page with a decorative eyebrow or slogan;
- do not skip heading levels;
- avoid going deeper than level 3;
- do not use bold paragraphs as fake headings;
- use a horizontal rule only for a genuine transition on a landing or hub page, not between
  every short section.

### 4.3 Prose

Use second person and active voice in learning material. Explain the purpose before the
mechanism.

The first paragraph should answer:

- what the page lets the reader do or understand;
- what it assumes;
- what observable result the reader will reach.

Preferred paragraph rhythm:

1. state the idea or task;
2. show the smallest complete example;
3. show the result;
4. interpret the result;
5. link to the next relevant detail.

Style rules:

- keep paragraphs focused on one idea;
- use backticks for code identifiers, commands, filenames, shapes, and literal values;
- use bold for a small number of conceptual anchors or interface labels;
- use italics sparingly for scientific emphasis;
- spell package names exactly as published: `BrainX`, `brainstate`, `braincell`,
  `BrainTrace`, `brainpy.state`, and `brainmass`;
- state units whenever a numerical value is physically meaningful;
- avoid promotional superlatives that are not supported by a measurement or comparison;
- do not narrate obvious UI instructions such as "click the blue button."

### 4.4 Links and cross-references

Use semantic Sphinx references for internal targets.

RST:

```rst
:doc:`installation guide <getting_started/installation>`
:class:`~brainmass.Simulator`
:func:`brainmass.viz.plot_timeseries`
:mod:`brainunit`
```

MyST:

```markdown
{doc}`/getting_started/installation`
{class}`~brainmass.Simulator`
{func}`brainmass.viz.plot_timeseries`
{mod}`brainunit`
```

Do not hardcode internal `.html` URLs. Use `:link-type: doc` for cards. Use a normal named
link for external destinations and identify external projects clearly.

### 4.5 Math

Use MyST dollar math in notebooks and Markdown:

```markdown
The time constant is $\tau$.

$$
\tau \frac{dV}{dt} = -(V - V_\mathrm{rest}) + RI.
$$
```

Use `:math:` and `.. math::` in RST. Introduce an equation in prose before displaying it,
then define every non-obvious symbol and interpret the consequence after it. Do not present a
wall of equations without connecting them to executable quantities.

### 4.6 Tables

Use tables for exact comparisons, parameter mappings, supported variants, and compact API
summaries. Use bullets for loose collections.

RST tables should normally use `list-table`:

```rst
.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Meaning
   * - ``dt``
     - Integration step, expressed as a time quantity.
```

Keep tables narrow enough for mobile. If a table needs more than four dense columns, split
it, shorten the cell text, or move detailed prose below it. Never rely on color alone to
communicate a table state.

## 5. Landing and Hub Pages

The contents and navigation groups are package-specific. Their presentation is not.

### 5.1 Landing page grammar

A package landing page should use this visual sequence:

1. literal package title;
2. one concise paragraph defining the package and its relationship to BrainX;
3. optional one-paragraph differentiator or important compatibility note;
4. optional installation tabs if installation is a primary first-page task;
5. a small navigation grid for the reader's next actions;
6. one compact runnable or static quick example when it materially helps orientation;
7. a short ecosystem/citation line;
8. hidden toctrees that drive the left sidebar.

This is a page-style sequence, not a required documentation taxonomy. Omit any element the
package does not need.

### 5.2 Navigation cards

Cards are the primary button-like control inside BrainX documentation. Use them for a small
set of peer destinations, not as wrappers around every paragraph.

Canonical pattern:

```rst
.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: :material-regular:`rocket_launch;2em` Get started
      :link: getting_started/installation
      :link-type: doc

      Install the package and verify the runtime.

   .. grid-item-card:: :material-regular:`hub;2em` Core concepts
      :link: concepts/index
      :link-type: doc

      Build the mental model used by the rest of the documentation.

   .. grid-item-card:: :material-regular:`data_exploration;2em` API reference
      :link: reference/index
      :link-type: doc

      Look up public classes, functions, parameters, and return values.
```

Card rules:

- use a responsive column declaration such as `1 2 3 3`;
- use one icon family, preferably Material icons supplied by `sphinx-design`;
- do not mix emoji, Material icons, and custom SVG icons in one grid;
- keep titles parallel in grammar and capitalization;
- keep the description to one or two short sentences;
- make the whole card the target with `:link:` and `:link-type: doc`;
- use external URL cards only when the destination is genuinely outside the docs;
- use a consistent gutter of `2` or `3` within the page;
- avoid shadows and borders that differ from neighboring card groups;
- do not put cards inside cards;
- do not use undefined custom classes such as `package-learn-card` unless the matching CSS
  is committed, tested, and necessary.

### 5.3 Buttons

The inspected BrainX docs do not use standalone rounded CTA buttons as a primary page
pattern. They use:

- clickable cards for navigation choices;
- tabs for mutually exclusive installation variants;
- the code-copy control supplied by `sphinx-copybutton`;
- icon controls owned by the shared header.

Preserve that behavior. Do not add raw HTML `<button>` elements to narrative pages.

When a single compact command truly needs button treatment, use a Sphinx Design button,
not custom HTML:

```rst
.. button-ref:: getting_started/quickstart
   :color: primary
   :outline:

   Continue to the quickstart
```

Use no more than one primary button in a local decision area. For three or more peer
destinations, use cards. For an inline contextual destination, use a text link.

Standalone button rules:

- let `sphinx-design` and the shared theme own color, border radius, hover, and focus states;
- use the outlined style by default and a filled primary style only for the clear next action;
- start the label with a verb and keep it to two to five words;
- do not add a trailing period;
- use a familiar leading icon only when it improves recognition;
- do not use icon-only content buttons;
- do not create a disabled navigation button; omit unavailable destinations or explain their
  status in prose;
- keep adjacent buttons equal in visual weight unless one action is genuinely primary.

### 5.4 Tabs

Tabs are for alternate forms of the same task, especially platform-specific installation.

```rst
.. tab-set::

   .. tab-item:: CPU

      .. code-block:: bash

         pip install -U "package[cpu]"

   .. tab-item:: NVIDIA GPU (CUDA 12)

      .. code-block:: bash

         pip install -U "package[cuda12]"

   .. tab-item:: NVIDIA GPU (CUDA 13)

      .. code-block:: bash

         pip install -U "package[cuda13]"

   .. tab-item:: TPU

      .. code-block:: bash

         pip install -U "package[tpu]"
```

Tab rules:

- each tab must complete the same task;
- label the environment precisely;
- keep commands directly inside the tab;
- do not hide prerequisites, warnings, or semantically different workflows in tabs;
- put shared explanation before or after the tab set rather than repeating it in every tab;
- on narrow screens, ensure labels can wrap without clipping.

### 5.5 Hidden toctrees

The left navigation is generated from hidden toctrees placed on the landing or relevant hub
page:

```rst
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: [navigation label]

   path/to/page
   path/to/another-page
```

Use human-readable captions, stable page order, and the shallowest `maxdepth` that exposes
useful navigation. Do not use hardcoded manual sidebar HTML.

## 6. Narrative and Executable Pages

### 6.1 When a page should be a notebook

Use a notebook when the displayed result is part of the explanation. Typical examples are:

- a simulation trace;
- a spike raster;
- a model representation;
- a parameter sweep;
- a gradient or loss check;
- a shape, key, unit, or state inspection;
- a performance comparison that is measured rather than asserted.

Use RST or Markdown for purely conceptual prose, navigation, installation commands, and API
catalogs.

### 6.2 Notebook page grammar

Use this repeated unit throughout a notebook:

1. Markdown states what the next step does and why.
2. One code cell performs one coherent step.
3. The stored output appears directly below that cell.
4. Markdown interprets what the reader should notice.

A strong notebook begins like this:

```markdown
# Page title

In this page you will [observable outcome]. By the end you will be able to:

1. [action],
2. [action], and
3. [action].

## Setup

We import [packages] and set [seed/environment/time step].
```

Then use numbered task headings when the sequence matters:

```markdown
## 1. Construct the model
## 2. Run the simulation
## 3. Inspect the intermediate state
## 4. Plot and interpret the result
```

End with a short summary and semantic links:

```markdown
## What you learned

- [verified fact]
- [verified fact]

## Next steps

- {doc}`/path/to/page` - [specific reason to read it].
```

### 6.3 Code cells

Code cells should be complete, readable, and reproducible.

- Put imports in one setup cell unless a later import is pedagogically meaningful.
- Set the physical environment explicitly, including `dt` when relevant.
- Set random seeds for stochastic examples.
- Use public APIs in user-facing pages.
- Prefer one logical operation per cell.
- Keep the typical cell under 30 lines; move reusable or distracting support code into a
  helper cell or package utility.
- Name quantities after their scientific meaning rather than `x1`, `tmp`, or `result2`.
- Include units in values and output labels.
- Avoid hidden state from cells the reader cannot see.
- Do not require the reader to execute cells out of order.
- Do not include secret tokens, personal paths, or machine-specific device assumptions.

### 6.4 Intermediate running results

Intermediate outputs are evidence. Display the smallest result that proves the preceding
step worked.

| Question being answered | Preferred output |
| --- | --- |
| Did the object initialize correctly? | compact object representation or selected public fields |
| Did the data have the expected structure? | labeled keys, shape, dtype, and unit |
| Did the simulation cover the expected interval? | first/last time and number of steps |
| Did optimization improve the model? | initial/final loss plus a short curve when useful |
| Did batching/vectorization work? | input and output shapes, optionally one selected slice |
| Did dynamics change? | labeled plot with the compared traces |
| Did two methods agree? | error metric with tolerance and a compact comparison plot |

Use labeled text rather than an unexplained value:

```python
print("recorded keys:", list(result))
print("V shape (steps, regions):", result["V"].shape)
print("time range:", result["ts"][0], "to", result["ts"][-1])
```

Good output:

```text
recorded keys: ['V', 'w', 'ts']
V shape (steps, regions): (2000, 1)
time range: 0.1 ms to 200.0 ms
```

Avoid:

- dumping a full array when shape and range answer the question;
- progress-bar residue in the committed notebook;
- import warnings that are unrelated to the lesson;
- device discovery logs;
- hundreds of training steps printed line by line;
- raw tracebacks in a successful tutorial;
- screenshots of terminal output;
- manually typed output in a Markdown code fence presented as if it were executed.

If an error is the lesson, catch it narrowly and print the exception type and message, or
use a purpose-built error demonstration. The final committed notebook must not contain an
uncaught `output_error` cell.

### 6.5 Plot outputs

Most BrainX notebook figures use a wide scientific aspect ratio; `figsize=(8, 4)` is the
most common baseline in the inspected sources. Use it unless the data needs a different
shape.

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(times, values, label="membrane potential")
ax.set(
    title="Response to a brief current pulse",
    xlabel="Time (ms)",
    ylabel="Voltage (mV)",
)
ax.legend()
fig.tight_layout()
```

Plot rules:

- show one principal claim per figure;
- label both axes and include physical units;
- use a descriptive takeaway title, not `Result` or `Plot 1`;
- include a legend only when there are multiple encodings;
- use consistent colors for the same scientific entities across a page;
- do not rely on red/green alone;
- keep line widths, markers, and labels legible at the rendered article width;
- shade stimuli or excluded transients lightly and label them;
- suppress redundant object representations after a plot, for example with a trailing
  semicolon;
- use a package plotting helper when it expresses the domain more clearly than raw
  Matplotlib;
- avoid interactive-only outputs unless a static fallback is stored and verified.

Use side-by-side subplots only for a real comparison. Do not compress four unrelated plots
into a dashboard merely to save vertical space.

### 6.6 Stored output policy

The standard is:

```python
nb_execution_mode = "off"
```

This means:

- the author executes the notebook before committing;
- outputs are stored in the `.ipynb` file;
- Sphinx renders those stored outputs;
- the normal HTML build does not rerun scientific workloads;
- CI separately verifies notebooks when the package's runtime allows it.

Before committing an executable page:

1. restart the kernel;
2. run all cells from top to bottom;
3. confirm execution counts are monotonic;
4. confirm there are no error outputs;
5. remove irrelevant warnings, logs, and progress bars;
6. inspect every text output for labels and units;
7. inspect every plot at documentation width;
8. save the notebook with the final outputs.

Never edit an output payload by hand. Rerun the cell that owns it.

### 6.7 Notebook metadata

Use notebook format 4 and a Python 3 kernel:

```json
{
  "kernelspec": {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
  },
  "language_info": {
    "name": "python"
  }
}
```

Do not depend on editor-specific cell metadata for essential rendering. Clear collapsed
states and transient widget metadata unless the published result requires them.

## 7. Images, Diagrams, and Other Visuals

### 7.1 What deserves an authored image

Use an authored image when spatial relationships carry the explanation better than prose:

- system architecture;
- biological structure;
- signal flow;
- memory or computational complexity comparison;
- a learning-path dependency map;
- a UI or external tool screenshot that the reader must identify visually.

Use a generated notebook plot for numerical results. Do not convert a real plot or console
output into a static screenshot.

### 7.2 Asset location and naming

Put authored package assets under:

```text
docs/_static/images/
```

Use lowercase descriptive filenames with hyphens, for example:

```text
alignpre-alignpost.png
memory-complexity.png
learning-path.svg
```

Keep package logos directly under `_static/` only when existing theme configuration expects
that path. Do not use ambiguous filenames such as `image1.png`, `new.png`, or `final2.png`.

### 7.3 Format and resolution

- use SVG for diagrams, dependency maps, and line-based schematics;
- use PNG for plots or raster art requiring lossless text and sharp edges;
- use WebP for large illustrative raster images where smaller delivery size matters;
- use JPEG only for photographic material without transparency;
- export raster diagrams at roughly 2x their intended display size;
- crop unused whitespace before committing;
- keep text inside a diagram readable at the final article width.

Do not embed multi-megabyte base64 images in Markdown. Notebook output images are the
exception because Jupyter owns their representation.

### 7.4 Placement

Place a figure immediately after the paragraph that introduces it. The next paragraph
should interpret the visual or state the takeaway.

Do not:

- put a figure before the reader knows why it matters;
- collect all figures at the end of a page;
- float a figure beside long code;
- use an image as decoration between sections;
- repeat the same figure at multiple sizes on one page.

### 7.5 RST figure pattern

Use `figure` when a caption adds meaning:

```rst
.. _fig-alignpre-alignpost:

.. figure:: /_static/images/alignpre-alignpost.png
   :alt: AlignPre places synapse state before the connection matrix on the
         presynaptic side; AlignPost places it after the matrix on the
         postsynaptic side.
   :width: 95%
   :align: center

   **AlignPre and AlignPost.** The location of synapse state determines whether
   memory scales with the presynaptic or postsynaptic population.
```

Use `image` only when no caption is needed, for example a DOI badge or a compact navigation
map already explained in adjacent prose.

### 7.6 MyST/notebook figure pattern

```markdown
:::{figure} /_static/images/alignpre-alignpost.png
:name: fig-alignpre-alignpost
:alt: AlignPre places synapse state before the connection matrix on the presynaptic side; AlignPost places it after the matrix on the postsynaptic side.
:width: 95%
:align: center

**AlignPre and AlignPost.** The location of synapse state determines whether memory scales
with the presynaptic or postsynaptic population.
:::
```

A simple Markdown image is acceptable for a small, unnumbered illustration:

```markdown
![Short but specific description](/_static/images/example.png)
```

Prefer the figure directive for any important scientific diagram because it supports a
caption, target, numbering, and cross-reference.

### 7.7 Width

Use width according to the information density:

- `95%` to `100%`: wide architecture or comparison diagram;
- `60%` to `80%`: a single mechanism or compact conceptual figure;
- fixed width around `400px` to `500px`: a narrow learning map, badge group, or small
  schematic that becomes awkward when stretched.

Always verify fixed-width figures on mobile. Never set both a rigid width and height that
distorts the aspect ratio.

### 7.8 Alt text and captions

Alt text describes the information needed by a reader who cannot see the image. A caption
states the conclusion or context.

Good alt text identifies:

- the layout or axes;
- the important elements;
- the relationship among those elements;
- the direction of a trend when the visual communicates one.

Avoid `alt="diagram"`, `alt="plot"`, filenames, and captions copied verbatim into alt text.

### 7.9 Remote images

Commit explanatory assets locally so builds remain reproducible. Remote images are allowed
for live badges whose value is maintained by the provider, such as a DOI badge; give them
an explicit target and alt label.

Do not hotlink tutorial figures from unrelated repositories. Download an appropriately
licensed copy, record attribution, and commit it under `_static/images/`.

## 8. Reusable Page Components

### 8.1 Admonitions

Use admonitions only when the information benefits from interruption.

| Type | Use |
| --- | --- |
| `note` | prerequisite, scope boundary, stored-output explanation |
| `tip` | optional shortcut or useful optimization |
| `important` | semantic constraint required for correct use |
| `warning` | likely failure, data loss, invalid scientific interpretation, or incompatible environment |

MyST:

```markdown
:::{important}
Build sparse connectivity before entering `jax.jit`; pass only changing values into the
compiled function.
:::
```

RST:

```rst
.. important::

   Build sparse connectivity before entering ``jax.jit``; pass only changing
   values into the compiled function.
```

Do not use an admonition for a normal definition, every section summary, or promotional
copy. Prefer a short paragraph when the reader does not need a visual interruption.

### 8.2 Dropdowns

Use a dropdown for optional long detail such as derivations, complete logs, or advanced
implementation notes. The core path must remain understandable while the dropdown is
closed.

```rst
.. dropdown:: Derivation

   [Optional derivation.]
```

Never hide prerequisites, primary results, safety warnings, or the only explanation of a
code cell in a dropdown.

### 8.3 Code blocks

Use an explicit lexer:

```rst
.. code-block:: bash

   python -m pip install -U "package[cpu]"
```

Use `python`, `bash`, `toml`, `yaml`, `json`, `text`, or another accurate language. Do not
label console output as Python. Keep shell prompts out of copyable commands.

### 8.4 Cards versus tables versus tabs

Choose by interaction:

- **card:** navigate to one of several peer destinations;
- **table:** compare exact properties across items;
- **tab:** switch among alternate versions of the same task;
- **dropdown:** reveal optional depth without breaking the main reading flow;
- **admonition:** interrupt for information with elevated importance;
- **text link:** continue naturally to one related destination.

Do not use a card merely to add a border around ordinary prose.

## 9. Code and Output Visual Style

### 9.1 Static snippets

Use static code blocks for:

- installation commands;
- configuration fragments;
- signatures or one isolated API pattern;
- pseudocode;
- a short example whose output is not part of the claim.

The copy button should be enabled globally through `sphinx-copybutton`. Do not add a custom
copy button to individual blocks.

### 9.2 Runnable examples

Use a notebook when execution produces evidence. Keep code and result together. Do not show
code in RST and then insert a screenshot of a result several paragraphs later.

### 9.3 Output hierarchy

Prefer outputs in this order:

1. a concise scalar or labeled structural check;
2. a domain-specific rich representation when it is informative;
3. a plot for trends, dynamics, or comparison;
4. a table for multiple exact values;
5. a long raw representation only when its structure is the lesson.

Text and plot outputs may coexist when the text verifies structure and the plot carries the
scientific result. Avoid duplicating the same information in both.

### 9.4 Reproducibility

Every executable page should state or encode:

- required package version when version-sensitive;
- seed for stochastic behavior;
- integration step and physical units;
- device assumptions if behavior differs by device;
- expected shape or tolerance for a correctness claim;
- manageable runtime and memory requirements.

Do not make a documentation build depend on downloading a large dataset. Use a small
committed fixture, a package dataset helper with caching, or a clearly optional cell.

## 10. API Documentation Style

### 10.1 API hub pages

Use RST, `currentmodule`, and `autosummary`. Group public symbols by meaning and add one
sentence explaining each group.

```rst
Public API
==========

.. currentmodule:: package

Core Types
----------

Objects used to construct and run a model.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Model
   Simulator

Utilities
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   load_data
   plot_result
```

Only document the public import path. Do not expose private implementation modules in
titles, links, or `viewcode` targets for re-exported public objects.

### 10.2 Autosummary templates

`docs/_templates/classtemplate.rst`:

```rst
.. role:: hidden
   :class: hidden-section

.. currentmodule:: {{ module }}

{{ name | underline }}

.. autoclass:: {{ name }}
   :members:
   :inherited-members:
```

`docs/_templates/functiontemplate.rst`:

```rst
.. role:: hidden
   :class: hidden-section

.. currentmodule:: {{ module }}

{{ name | underline }}

.. autofunction:: {{ name }}
```

Use one owner for each generated target. Do not manually repeat member sections already
emitted by `:members:`.

### 10.3 Docstring profile

The canonical profile is NumPy-style docstrings with types documented in the docstring and:

```python
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "none"
```

Example:

```python
def simulate(model, duration, *, dt):
    """Run a model and record its public state.

    Parameters
    ----------
    model : Module
        Initialized model to advance.
    duration : Quantity
        Physical simulation duration.
    dt : Quantity
        Integration step. Must have time dimensions.

    Returns
    -------
    dict[str, Array | Quantity]
        Recorded time axis and state trajectories. The leading axis is time.

    Raises
    ------
    ValueError
        If ``duration`` is not positive or is not compatible with ``dt``.

    Notes
    -----
    The model's final state remains installed after the call.

    Examples
    --------
    >>> result = simulate(model, 100.0 * u.ms, dt=0.1 * u.ms)
    >>> result["V"].shape[0]
    1000
    """
```

Docstring rules:

- start with a one-line imperative or factual summary;
- explain behavior and semantics, not implementation trivia;
- give parameter units, accepted shapes, defaults, and state mutations;
- describe each return value, shape, unit, and ordering;
- document meaningful exceptions;
- use `Notes` for mathematical or lifecycle semantics;
- make examples minimal, public, and executable;
- put references in a `References` section with enough bibliographic information to find
  the source;
- do not repeat the signature in prose.

Existing packages may use a type-hint-first profile during migration:

```python
extensions += ["sphinx_autodoc_typehints"]
autodoc_typehints = "description"
```

In that profile, do not duplicate the same types in a Napoleon-rendered field list. Choose
one profile for the entire package. Mixed ownership causes duplicate fields and Sphinx
warnings.

### 10.4 API examples versus tutorials

An API example should show the smallest valid call and result. A tutorial should explain a
workflow and interpret intermediate results. Do not turn each API entry into a mini tutorial,
and do not make a tutorial serve as the only parameter reference.

## 11. Responsive and Accessibility Requirements

Every new or changed page must work at desktop and mobile widths.

### 11.1 Required checks

- card grids collapse to one column without clipped titles;
- tab labels wrap or remain scrollable without covering content;
- code blocks scroll horizontally rather than widening the article;
- tables remain usable or are simplified;
- images scale down and preserve aspect ratio;
- fixed-width images do not overflow;
- long API signatures wrap or scroll within their component;
- header dropdowns and the sidebar remain keyboard-operable;
- interactive targets are at least `44px` on touch surfaces, with the shared header using
  `48px` for mobile dropdown controls;
- focus state is visible;
- headings retain a logical order;
- each meaningful image has alt text;
- color is never the only indicator;
- animation respects `prefers-reduced-motion`.

### 11.2 Content accessibility

Explain abbreviations on first use. Use precise link labels such as "installation guide"
rather than "here." Avoid symbol-only legends. For plots, label lines directly or provide a
clear legend and describe the important trend in surrounding text.

## 12. Development Workflow

### 12.1 Environment setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-doc.txt
python -m pip install -e .
```

Use the Python version supported by the package and its CI matrix. Do not hardcode a newer
documentation Python version than the package can import under.

### 12.2 Authoring a notebook

```bash
jupyter lab docs/path/to/page.ipynb
```

Before review, perform a clean execution. One practical command is:

```bash
jupyter nbconvert \
  --to notebook \
  --execute docs/path/to/page.ipynb \
  --inplace \
  --ExecutePreprocessor.timeout=200
```

Then open the notebook and visually inspect every output. Automated execution proves that
cells run; it does not prove that a plot is legible or that the output is pedagogically
useful.

### 12.3 Local HTML build

Use a warning-strict build during development:

```bash
python -m sphinx \
  -W --keep-going \
  -b html \
  docs \
  docs/_build/html
```

Open:

```text
docs/_build/html/index.html
```

For a clean rebuild:

```bash
rm -rf docs/_build
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The deletion target above is intentionally the explicit build directory. Never use a broad
or unresolved path in cleanup commands.

### 12.4 Shared-header development

The extension normally fetches and caches canonical assets from:

```text
https://brainx.chaobrain.com/shared-header
```

To preview an unpublished shared-header change against a package build:

```bash
BRAINX_HEADER_LOCAL=/absolute/path/to/dist/shared-header \
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

For a release build, force a fresh canonical fetch:

```bash
BRAINX_HEADER_TTL=0 \
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Offline builds may set `BRAINX_HEADER_OFFLINE=1` only after a complete cache exists.

### 12.5 Additional validation

Run relevant checks before merge:

```bash
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
python -m sphinx -W --keep-going -b doctest docs docs/_build/doctest
```

Use `doctest` only when enabled and supported by the package. Also run the package tests that
cover APIs demonstrated in the documentation.

Search the built log for:

- missing references;
- duplicate object descriptions;
- documents not included in a toctree;
- field-list formatting warnings;
- missing static assets;
- unknown directives or roles;
- notebooks with unsupported MIME output.

### 12.6 CI build pattern

The documentation job should:

1. check out the repository;
2. use a supported Python version;
3. install system dependencies such as Pandoc;
4. install runtime, documentation, and package dependencies;
5. execute the selected notebook validation set;
6. build HTML with warnings treated as errors;
7. assert that `_build/html/index.html` exists;
8. upload the built HTML as an artifact;
9. deploy only the verified artifact.

Minimal build step:

```yaml
- name: Build documentation
  env:
    BRAINX_HEADER_TTL: "0"
  run: |
    python -m sphinx -W --keep-going -b html docs docs/_build/html
    test -f docs/_build/html/index.html
```

The current BrainX deployment pattern uses immutable timestamped releases and atomically
switches a `current` symlink after verifying `index.html`. Preserve that behavior for hosted
package docs; do not upload directly over the live directory.

### 12.7 Read the Docs configuration

```yaml
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "[supported version]"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - requirements: requirements-doc.txt
```

## 13. Review Checklist

### 13.1 Every page

- [ ] The title is literal, unique, and sentence-cased.
- [ ] The opening states the page outcome and assumptions.
- [ ] Heading hierarchy is shallow and valid.
- [ ] Code names use inline code formatting.
- [ ] Physical values include units.
- [ ] Internal links use semantic Sphinx roles.
- [ ] No internal link hardcodes `.html`.
- [ ] Admonitions are rare and correctly typed.
- [ ] The page has no raw HTML for standard Sphinx components.
- [ ] The right-hand table of contents is useful at level 2.
- [ ] The page works at mobile width.

### 13.2 Cards, tabs, and controls

- [ ] Cards represent peer destinations and have whole-card links.
- [ ] Card titles and descriptions fit without clipping.
- [ ] One icon family is used consistently.
- [ ] Tabs represent alternate forms of one task.
- [ ] There is at most one primary standalone button in a local decision area.
- [ ] Copy controls come from `sphinx-copybutton`.
- [ ] No package page reimplements the BrainX header controls.

### 13.3 Notebooks and outputs

- [ ] The kernel was restarted and all cells were run top to bottom.
- [ ] Execution counts are monotonic.
- [ ] There are no uncaught error outputs.
- [ ] Random behavior has a fixed seed.
- [ ] Intermediate text output is labeled and compact.
- [ ] Arrays are summarized rather than dumped.
- [ ] Warnings, device logs, and progress residue are removed.
- [ ] Plots have a takeaway title, axis labels, and units.
- [ ] Plot colors remain distinguishable without red/green discrimination.
- [ ] Stored outputs are committed.
- [ ] `nb_execution_mode` remains `off` for the Sphinx build.

### 13.4 Images

- [ ] The image is necessary to the explanation.
- [ ] It is introduced before it appears and interpreted after it.
- [ ] Explanatory assets are local and have clear filenames.
- [ ] Format matches the visual type.
- [ ] Raster resolution is sufficient at the displayed width.
- [ ] Alt text conveys the relevant information.
- [ ] Important figures have captions and cross-reference names.
- [ ] Fixed widths do not overflow on mobile.

### 13.5 API pages

- [ ] Only public import paths are documented.
- [ ] Autosummary targets are generated once.
- [ ] Docstrings use one package-wide style profile.
- [ ] Parameter units, shapes, defaults, and mutations are documented.
- [ ] Return shapes and units are documented.
- [ ] Examples use public APIs and run successfully.
- [ ] The strict Sphinx build has no duplicate-object or field-list warnings.

### 13.6 Build and release

- [ ] Documentation dependencies are declared in `requirements-doc.txt`.
- [ ] The package imports under the documentation Python version.
- [ ] A clean warning-strict HTML build passes.
- [ ] Link checking has been run for changed links.
- [ ] The built landing page, one notebook, and one API page were visually inspected.
- [ ] Shared header/footer assets render and mobile navigation works.
- [ ] The deployment switches atomically to a verified artifact.

## 14. Legacy Drift Not to Copy

The inspected repositories contain older patterns that should not become new conventions:

- emoji mixed with Material icons in navigation cards;
- package-specific card classes with no committed stylesheet;
- raw `<center>` and `<img>` markup in notebook Markdown;
- hotlinked explanatory images from other documentation sites;
- both `myst_nb` and `nbsphinx` enabled in the same build;
- `sphinx_thebe` configured against an unrelated example repository without a tested live
  execution feature;
- broad warning suppression instead of fixing references;
- configuration code that deletes build outputs or generated API trees at import time;
- Google-style authoring guidance in a package whose current source predominantly uses
  NumPy-style docstrings;
- duplicated API type information from both Napoleon and
  `sphinx-autodoc-typehints`;
- notebook outputs containing GPU discovery warnings, progress-bar residue, or uncaught
  errors;
- manually linked internal `.html` files instead of semantic document links;
- a declared custom CSS file that is not present in `_static`.

When maintaining an old page, do not rewrite it solely for stylistic purity. Apply this
standard when the page is materially edited, and fix drift that affects correctness,
accessibility, responsiveness, or build reliability immediately.

## 15. Handoff Contract for Future Documentation Work

Once this standard is adopted, a content request only needs to specify:

```text
Package:
Page path:
Page title:
Audience and prerequisite:
What the page must teach or prove:
Required sections or facts:
Required APIs/equations:
Expected executable evidence:
Required authored figures or datasets:
Links to related pages:
```

Everything else defaults to this guide:

- the BrainX header/footer and responsive behavior;
- Sphinx Book Theme layout;
- heading, prose, and cross-reference style;
- cards, tabs, buttons, tables, and admonitions;
- image placement, width, caption, and alt-text rules;
- code-cell and intermediate-output presentation;
- notebook execution and storage policy;
- API rendering and docstring profile;
- local build, validation, CI, and deployment behavior.

That separation is intentional: authors decide what a package needs to say; the ecosystem
standard decides how it should look, behave, and be verified.
