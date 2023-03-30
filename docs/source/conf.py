# Configuration file for the Sphinx documentation builder.
#
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MapPy'
copyright = '2023, Yuu Miino'
author = 'Yuu Miino'

from mappy import __version__
version = __version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx_design'
]

autosummary_generate=True

autodoc_typehints = "none"
autodoc_default_options = {
    'inherited_members': False,
    'show-inheritance': False
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'styles/MapPy.css',
]

html_theme_options = {
    "logo": {
        "text": "MapPy package",
        "image_light": "https://miino.sytes.net/figs/089-traj.gif",
        "image_dark": "https://miino.sytes.net/figs/089-traj.gif",
    },
    "switcher": {
        "version_match": version,
        "json_url": "https://yuu-miino.github.io/doc.mappy/latest/_static/switcher.json",
    },
    "navbar_start": ["navbar-logo", "version-switcher"]
}

html_list = ["search-field.html", "sidebar-nav-bs.html"]
html_sidebars = {
    "contents/**": html_list,
}

mathjax3_config = {
    'tex': {
        'macros': {
            'bm': [r"\boldsymbol{#1}", 1],
            'deriv': [r"\frac{d#1}{d#2}", 2],
            'pderiv': [r"\frac{\partial#1}{\partial#2}", 2],
            'parens': [r"\left(#1\right)", 1],
            'braces': [r"\left\{#1\right\}", 1],
            'brackets': [r"\left[#1\rigth]", 1],
            'set': [r"\braces{#1}", 1],
            'case': [r"\left{#1\right.", 1],
            'R': [r"\mathbb{R}"]
        }
    }
}

# -- Options for sphinx-multiversion -----------------------------------------
# smv_tag_whitelist = r'v^\d+\.\d+$'
# smv_branch_whitelist = r'^master$'
