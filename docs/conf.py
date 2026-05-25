# CompileIQ Documentation Configuration File

import os
import sys
from datetime import datetime

# Add Python packages to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "CompileIQ"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

release = os.environ.get("DOC_VERSION", os.environ.get("SPHINX_CIQ_VER", "latest"))
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

docs_root_url = (
    os.environ.get(
        "CIQ_DOCS_BASE_URL",
        "https://nvidia.github.io/CompileIQ/",
    ).rstrip("/")
    + "/"
)

switcher_version = os.environ.get("DOC_VERSION", release)

if switcher_version == "latest":
    docs_version_folder = "latest"
elif switcher_version.startswith("v"):
    docs_version_folder = switcher_version
else:
    docs_version_folder = f"v{switcher_version}"

html_baseurl = f"{docs_root_url}{docs_version_folder}/"
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
        "json_url": f"{docs_root_url}versions.json",
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


def setup(app):
    if os.path.exists("_static/custom.css"):
        app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", autodoc_skip_enum_members)
