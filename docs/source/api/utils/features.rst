Features
========

Feature engineering utilities and decorators.

.. currentmodule:: unravel.utils.features

The features module provides built-in feature functions and a decorator for creating custom
graph features.

.. autofunction:: graph_feature

.. autofunction:: x_normed
.. autofunction:: y_normed
.. autofunction:: speeds_normed
.. autofunction:: velocity_components_2d_normed

.. autofunction:: distance_normed
.. autofunction:: angle_features

.. code-block:: python

   from unravel.utils.features import graph_feature

   @graph_feature(
       cols=["x", "y"],
       returns=["distance_from_center"],
       type="node"
   )
   def distance_from_center(x, y):
       import polars as pl
       return [pl.sqrt(x**2 + y**2)]

   # Use in converter
   from unravel.soccer import SoccerGraphConverter

   converter = SoccerGraphConverter(
       dataset=polars_dataset,
       node_feature_cols=[distance_from_center],
   )
