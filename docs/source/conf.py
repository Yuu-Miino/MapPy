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

def get_version_number(st):
    import re

    """
    A function that extracts the version number from a string that is enclosed in single or double quotes.
    by ChatGPT

    Parameters
    ----------
    st : str
        The input string containing the version number enclosed in quotes.

    Returns
    -------
    str or None
        The extracted version number as a string. If no version number is found, returns None.

    Examples
    --------
    >>> get_version_number("version = '0.0.4'")
    '0.0.4'
    >>> get_version_number('version = "0.0.5"')
    '0.0.5'
    """
    pattern = r"(?<=['\"])[^'\"]+(?=['\"])"  # Regular expression pattern
    match = re.search(pattern, st)  # Get the match for the regular expression in the string
    if match is None:
        return None  # Return None if no match is found
    return match.group()  # Return the matched substring

version_file = '../../mappy/_version.py'
if os.path.exists(version_file):
    with open(version_file) as f:
        version = get_version_number(f.read())
        if version is None:
            version = '0.0.1'
else:
    version = '0.0.1'
release = version
with open('out.dat', 'a') as f:
    f.write('1 '+version+str(type(version))+'\n')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.autosummary',
#    'sphinx_automodapi.automodapi',
    'sphinx.ext.mathjax',
    'sphinx_design',
    'sphinx_multiversion',
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
        "version_match": "current_version.name",
        "json_url": "https://yuu-miino.github.io/MapPy/master/_static/switcher.json",
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
            'case': [r"\left{#1\right.", 1]
        }
    }
}

# -- Options for sphinx-multiversion -----------------------------------------
# smv_tag_whitelist = r'v^\d+\.\d+$'
# smv_branch_whitelist = r'^master$'
