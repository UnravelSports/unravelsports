Graphs
======

Converting NFL tracking data to graph structures.

.. currentmodule:: unravel.american_football

.. autoclass:: AmericanFootballGraphConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.american_football import AmericanFootballGraphConverter

   converter = AmericanFootballGraphConverter(
       dataset=bdb_dataset,
       self_loop_ball=True,
       adjacency_matrix_connect_type="ball",
       adjacency_matrix_type="split_by_team",
       label_type="binary",
   )

   graphs = converter.to_pytorch_graphs()
