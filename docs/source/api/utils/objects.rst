Objects
=======

Base classes and core data structures.

.. currentmodule:: unravel.utils

.. autoclass:: GraphDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.utils import GraphDataset

   # Create dataset
   dataset = GraphDataset(graphs=graphs, format="pyg")

   # Split data
   train, test, val = dataset.split_test_train_validation(4, 1, 1)
