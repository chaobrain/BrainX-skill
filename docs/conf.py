from pathlib import Path
import os
import shutil
import sys


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
sys.path.insert(0, str(REPO_DIR))

project = "BrainX Skill"
author = "BrainX Developers"
copyright = "2026, BrainX Ecosystem"
release = "1.0.12"

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
html_css_files = ["brainx-skill.css"]
html_js_files = ["brainx-skill.js"]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_class_signature = "separated"
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
myst_enable_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]
exclude_patterns = [
    "_build",
    "_static/**",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "jupyter_execute/**",
]

html_theme = "sphinx_book_theme"
html_title = "BrainX Skill"
html_logo = "images/image.png"
html_favicon = html_logo
html_baseurl = "https://brainx.chaobrain.com/skill/"
html_copy_source = True
html_sourcelink_suffix = ""
html_last_updated_fmt = ""
html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/chaobrain/BrainX-skill",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_fullscreen_button": False,
}

nb_execution_mode = "off"
nb_execution_timeout = 200

# Keep local previews self-contained. Release builds may opt into the canonical
# production base URL with BRAINX_DOCS_PRODUCTION=1.
brainx_inject_base = os.environ.get("BRAINX_DOCS_PRODUCTION") == "1"


def _copy_brainx_header_logo(app, exception):
    """Expose the logo at the absolute path used by the shared BrainX shell."""
    if exception is not None or app.builder.format != "html":
        return

    source = DOCS_DIR / "images" / "brainx-ecosystem.webp"
    destination = Path(app.outdir) / "images" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def setup(app):
    app.connect("build-finished", _copy_brainx_header_logo)
