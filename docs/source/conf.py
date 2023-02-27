# Configuration file for the Sphinx documentation builder.
#
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HDS'
copyright = '2023, Yuu Miino'
author = 'Yuu Miino'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
#    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_design'
]

# autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = [
    'styles/hds.css',
]

html_theme_options = {
    "logo": {
        "text": "HDS package",
        "image_light": "https://miino.sytes.net/figs/089-traj.gif",
        "image_dark": "https://miino.sytes.net/figs/089-traj.gif",
    }
}

html_sidebars = {
    "contents/**": ["search-field.html", "sidebar-nav-bs.html"],
    "modules/*": ["search-field.html", "sidebar-nav-bs.html"],
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
            'case': [r"\left{#1\right.", 1]
        }
    }
}