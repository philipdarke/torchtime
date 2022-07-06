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
    "sphinx.ext.doctest",
    "sphinx_copybutton",
]

# sphinx.ext.autodoc settings
autodoc_member_order = "bysource"
exclude_patterns = ["_build"]

# sphinx_copybutton settings
copybutton_prompt_text = ">>> "
copybutton_line_continuation_character = "\\"
copybutton_image_svg = """
<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-file" width="44" height="44" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
  <path d="M14 3v4a1 1 0 0 0 1 1h4" />
  <path d="M17 21h-10a2 2 0 0 1 -2 -2v-14a2 2 0 0 1 2 -2h7l5 5v11a2 2 0 0 1 -2 2z" />
</svg>
"""  # noqa: E501

# HTML output
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
}
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
