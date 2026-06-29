formula_detection
=================

Python tooling to detect formulaic language use in historic documents.

``formula_detection`` started as a frequency-based phrase search over
tokenized text corpora, and has grown to support orthographic variant
detection, semantic clustering of formulaic phrases, phrase co-occurrence
and motif mining, document/element boundary detection in streams with
unknown boundaries, and an interactive workbench for categorising
formulaic phrases into user-defined types.

Installation
------------

.. code-block:: bash

   pip install formula_detection

The package depends on `fuzzy-search <https://pypi.org/project/fuzzy-search/>`_
for tokenization utilities, and on ``scikit-learn``, ``scipy``, ``networkx``,
and ``ipywidgets`` for the clustering, graph, and interactive-workbench
features.

Getting started
----------------

The :doc:`Formula-Detection-Usage` notebook is the place to start: it
walks through tokenization, term and co-occurrence frequencies, and basic
phrase extraction with ``FormulaSearch``.

For a beginner-friendly, step-by-step path aimed at historians with no
data-science background, see the seven-part tutorial series under
:doc:`tutorials/index`.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Usage guide

   Formula-Detection-Usage
   Usage

.. toctree::
   :maxdepth: 1
   :caption: Document pattern detection

   Document-Pattern-Detection
   Motif-Based-Boundary-Detection
   Phrase-Recurrence-Scale-Grouping
   Phrase-Typing-Workbench
   Formulaic-Language-Over-Time
   Document-Pattern-Detection-Findings

.. toctree::
   :maxdepth: 1
   :caption: Applied to the resolutions corpus

   Document-Pattern-Detection-resolutions
   Motif-Based-Boundary-Detection-resolutions
   Phrase-Recurrence-Scale-Grouping-adjusted

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
