.. unravelsports documentation master file

unravelsports Documentation
============================

.. image:: https://github.com/UnravelSports/unravelsports.github.io/blob/main/imgs/unravelsports-5500x800.png?raw=true
   :alt: unravelsports logo
   :align: center

|

.. image:: https://img.shields.io/badge/powered%20by-UnravelSports-orange.svg?style=flat&colorB=E6B611&colorA=C3C3C3
   :target: https://unravelsports.github.io/
   :alt: UnravelSports

.. image:: https://img.shields.io/badge/license-Mozilla%20Public%20License%20v2.0-orange.svg?style=flat&colorA=C3C3C3&colorB=E20E6A
   :target: https://www.tldrlegal.com/license/mozilla-public-license-2-0-mpl-2
   :alt: License

|

The **unravelsports** package aims to aid researchers, analysts and enthusiasts by providing
intermediary steps in the complex process of converting raw sports data into meaningful
information and actionable insights.

Installation
------------

.. code-block:: bash

   pip install unravelsports

Features
--------

This package currently supports:

* ⚽ 🏈 **Polars DataFrame Conversion** - Convert tracking data to Polars DataFrames
* ⚽ 🏈 **Graph Neural Network** Training, Graph Conversion and Prediction
* ⚽ **Pressing Intensity** - Compute pressing intensity metrics
* ⚽ **Formation and Position Identification (EFPI)** - Elastic Formation and Position Identification

Quick Links
-----------

* `GitHub Repository <https://github.com/unravelsports/unravelsports>`_
* `Report Issues <https://github.com/unravelsports/unravelsports/issues>`_
* `UnravelSports Website <https://unravelsports.github.io/>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/soccer_gnn
   tutorials/pressing_intensity
   tutorials/formation_detection
   tutorials/american_football

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api/classifiers
   api/soccer
   api/american_football
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   additional/citations
   additional/license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
