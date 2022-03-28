from importlib.metadata import version as get_version

# Project information
project = "torchtime"
copyright = "2022, Philip Darke"
author = "Philip Darke"
version = get_version("torchtime")
release = version

# General
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build"]

# HTML output
html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,
    "github_user": "philipdarke",
    "github_repo": "torchtime",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
    "source_suffix": ".md",
}
html_title = "torchtime"
html_static_path = ["_static"]
html_extra_path = ["README.md"]
html_css_files = ["custom.css"]
pygments_style = "friendly"
html_show_sphinx = False
