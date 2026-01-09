Helpers
=======

Helper functions for data manipulation.

.. currentmodule:: unravel.utils

.. autofunction:: dummy_labels
.. autofunction:: add_dummy_label_column

.. autofunction:: dummy_graph_ids
.. autofunction:: add_graph_id_column

.. code-block:: python

   from unravel.utils import add_dummy_label_column

   # Add random binary labels
   dataset.dataset = add_dummy_label_column(dataset.dataset)

Adding Graph IDs
~~~~~~~~~~~~~~~~

.. code-block:: python

   from unravel.utils import add_graph_id_column

   # Each frame is a separate graph
   dataset.dataset = add_graph_id_column(dataset.dataset, by=["frame_id"])

   # Group by possession
   dataset.dataset = add_graph_id_column(
       dataset.dataset,
       by=["ball_owning_team_id", "period_id"]
   )
