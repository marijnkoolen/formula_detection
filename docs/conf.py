"""Sphinx configuration for the formula_detection documentation site."""
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'formula_detection'
copyright = '2026, Marijn Koolen'
author = 'Marijn Koolen'

try:
    from formula_detection import __version__ as release
except ImportError:
    release = '0.0.0'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',
    'nbsphinx',
]

# Napoleon: this codebase uses Google-style docstrings throughout.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Autosummary: auto-generate one stub page per module/class/function,
# recursing through all submodules, so new modules need no manual rst edits.
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

# nbsphinx: render notebooks using their already-saved outputs rather than
# re-executing them at build time. Several notebooks in this documentation
# run real, multi-million-token corpora and take minutes to execute; CI/RTD
# build environments shouldn't have to repeat that work (or have the
# underlying data files available) just to build the docs site.
nbsphinx_execute = 'never'

# MyST: render the existing .md documentation files (e.g. the design-notes
# findings doc) alongside the .rst pages.
myst_enable_extensions = ['colon_fence']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

templates_path = ['_templates']
exclude_patterns = [
    '_build', '**.ipynb_checkpoints', 'Thumbs.db', '.DS_Store',
    # Superseded by the docstrings added directly to the source modules.
    'search_doc.md', 'vocabulary_doc.md',
    # A one-off data-download utility notebook, not part of the documentation.
    'download-Gutenberg-titles.ipynb',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] if os.path.isdir(os.path.join(os.path.dirname(__file__), '_static')) else []
