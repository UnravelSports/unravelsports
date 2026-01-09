Graphs
======

Converting soccer tracking data to graph structures for GNN training.

.. currentmodule:: unravel.soccer

.. autoclass:: SoccerGraphConverter
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.soccer import SoccerGraphConverter

   converter = SoccerGraphConverter(
       dataset=polars_dataset,
       self_loop_ball=True,
       adjacency_matrix_connect_type="ball",
       adjacency_matrix_type="split_by_team",
       label_type="binary",
   )

   # Convert to PyTorch Geometric
   graphs = converter.to_pytorch_graphs()

   # Or Spektral (deprecated)
   graphs = converter.to_spektral_graphs()
