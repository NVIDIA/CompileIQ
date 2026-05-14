# CompileIQ Documentation Configuration File

import os
import shutil
import sys
from datetime import datetime

# Add Python packages to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "CompileIQ"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

release = os.environ.get("SPHINX_CIQ_VER", "unstable")
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.mermaid",
]

# Add support for .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/__pycache__",
    "*.pyc",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

html_logo = "_static/nvidia-logo.png"

html_baseurl = (
    os.environ.get(
        "CIQ_DOCS_BASE_URL",
        "https://miniature-adventure-5v8nrk3.pages.github.io/",
    ).rstrip("/")
    + "/"
)

switcher_version = os.environ.get("SPHINX_MULTIVERSION_VERSION", "main")
html_context = {
    "base_url": html_baseurl.rstrip("/"),
}
local_preview = os.environ.get("CIQ_DOCS_LOCAL_PREVIEW") == "1"
navbar_end = ["navbar-icon-links"] if local_preview else ["theme-switcher", "navbar-icon-links"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/CompileIQ",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": navbar_end,
    "check_switcher": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "sidebar_includehidden": True,
    "collapse_navigation": False,
}

if not local_preview:
    html_theme_options["switcher"] = {
        "json_url": f"{html_baseurl}switcher.json",
        "version_match": switcher_version,
    }


html_static_path = ["_static"] if os.path.exists("_static") else []

html_title = "CompileIQ Documentation"

# -- Options for extensions --------------------------------------------------

# Pydantic
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_signature_prefix = "class"

autodoc_pydantic_model_members = False
autodoc_pydantic_settings_members = False
autodoc_pydantic_settings_undoc_members = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_default = True
autodoc_pydantic_field_show_required = True
autodoc_pydantic_field_show_optional = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_type_hints = "description"

# Multiversion configuration
smv_branch_whitelist = r"^main$"
smv_tag_whitelist = r"^v?(\d+!)?\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$"
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/tags/.*$"
smv_outputdir_format = "{ref.name}"
smv_latest_version = "main"
smv_prefer_remote_refs = True

# Local previews use sphinx-build against the live worktree, where
# sphinx-multiversion's ``versions`` template variable is unavailable.
html_additional_pages = {} if local_preview else {"switcher.json": "switcher.json"}

# Set Python domain primary
primary_domain = "py"

# Mock imports for optional dependencies not installed in the docs environment
autodoc_mock_imports = [
    "mlflow",
    "starlette",
]

# External links configuration
extlinks = {
    "github": (
        "https://github.com/NVIDIA/CompileIQ/blob/main/%s",
        "%s",
    ),
}

# Config copybutton
copybutton_prompt_text = ">>> |$ |# "
autosummary_generate = True
autoclass_content = "class"


def autodoc_skip_enum_members(app, what, name, obj, skip, options):
    from enum import Enum

    if isinstance(obj, Enum):
        return True
    return skip


def on_build_finished(app, exception):
    if exception:
        return
    root = os.path.dirname(app.outdir)
    src = os.path.join(app.outdir, "switcher.json.html")
    dst = os.path.join(root, "switcher.json")
    if os.path.exists(src):
        shutil.copy2(src, dst)

    # Ensuring the pages root redirects to the latest version
    index_path = os.path.join(root, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write(
                f"<!DOCTYPE html><html><head>"
                f'<meta http-equiv="refresh" content="0; url=./{smv_latest_version}/">'
                f'<link rel="canonical" href="./{smv_latest_version}/">'
                f"</head><body>"
                f'<p>Redirecting to <a ""href="./{smv_latest_version}/">{smv_latest_version}</a>...</p>'  # noqa
                f"</body></html>\n"
            )


def setup(app):
    if os.path.exists("_static/custom.css"):
        app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", autodoc_skip_enum_members)
    app.connect("build-finished", on_build_finished)
