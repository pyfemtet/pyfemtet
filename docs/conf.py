# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess
import shutil
project_root_path = os.path.abspath('..')
sys.path.insert(0, project_root_path)


# -- Project information -----------------------------------------------------

project = "Basic Sphinx Example Project"
copyright = "2022, Read the Docs core team"
author = "Read the Docs core team"


# -- General configuration ---------------------------------------------------
# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]



def setup(_):  # 勝手に走る関数らしい
    # sphinx-apidoc -f -o docs pyfemtet
    docs_path = os.path.split(__file__)[0]
    out_path = os.path.join(docs_path, "modules")
    package_root_path = os.path.join(project_root_path, 'pyfemtet')
    subprocess.run(['sphinx-apidoc', '-f', '-o', out_path, package_root_path])  # , cwd=


    source_file = os.path.join(project_root_path, "README.md")
    destination_file = os.path.join(docs_path, "README.md")

    shutil.copy2(source_file, destination_file)


# setup()
