Soccer
======

Soccer-specific functionality for tracking data analysis.

.. currentmodule:: unravel.soccer

The soccer module provides tools for loading, processing, and analyzing soccer tracking data,
including dataset conversion, graph creation, and tactical models.

.. toctree::
   :maxdepth: 2

   soccer/dataset
   soccer/graphs
   soccer/models

* :class:`~unravel.soccer.KloppyPolarsDataset` - Convert Kloppy data to Polars
* :class:`~unravel.soccer.SoccerGraphConverter` - Convert tracking data to graphs
* :class:`~unravel.soccer.PressingIntensity` - Compute pressing intensity
* :class:`~unravel.soccer.EFPI` - Formation and position detection
