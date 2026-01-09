Quick Start
===========

This guide will walk you through the basic workflow of using unravelsports.

Soccer: Train a GNN
-------------------

Step 1: Load Data with Kloppy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load your tracking data using Kloppy. Here we'll use the DFL open data:

.. code-block:: python

   from kloppy import sportec
   from unravel.soccer import KloppyPolarsDataset

   # Load tracking data
   kloppy_dataset = sportec.load_open_tracking_data(
       only_alive=True,
       limit=500
   )

Step 2: Convert to Polars DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert the Kloppy dataset to a Polars DataFrame:

.. code-block:: python

   # Convert to Polars format
   polars_dataset = KloppyPolarsDataset(
       kloppy_dataset=kloppy_dataset
   )

   # View the data
   print(polars_dataset.dataset.head())

The resulting DataFrame includes:

* Player positions (x, y, z coordinates)
* Velocities (vx, vy, vz)
* Accelerations (ax, ay, az)
* Team information
* Ball state and position
* Timestamps and frame IDs

Step 3: Convert to Graphs for GNN Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert the tracking data to graph structures for training Graph Neural Networks:

.. code-block:: python

   from unravel.soccer import SoccerGraphConverter
   from unravel.utils import add_dummy_label_column, add_graph_id_column

   # Add labels and graph IDs
   polars_dataset.dataset = add_dummy_label_column(polars_dataset.dataset)
   polars_dataset.dataset = add_graph_id_column(
       polars_dataset.dataset,
       by=["frame_id"]
   )

   # Create graph converter
   converter = SoccerGraphConverter(
       dataset=polars_dataset,
       self_loop_ball=True,
       adjacency_matrix_connect_type="ball",
       adjacency_matrix_type="split_by_team",
       label_type="binary",
   )

   # Convert to PyTorch Geometric graphs
   graphs = converter.to_pytorch_graphs()

Step 4: Train a Graph Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Split the data and train a model:

.. code-block:: python

   from unravel.utils import GraphDataset
   from unravel.classifiers import PyGLightningCrystalGraphClassifier
   import pytorch_lightning as pyl
   from torch_geometric.loader import DataLoader

   # Create dataset and split
   dataset = GraphDataset(graphs=graphs, format="pyg")
   train, test, val = dataset.split_test_train_validation(4, 1, 1)

   # Create data loaders
   train_loader = DataLoader(train, batch_size=32, shuffle=True)
   val_loader = DataLoader(val, batch_size=32)
   test_loader = DataLoader(test, batch_size=32)

   # Initialize model
   model = PyGLightningCrystalGraphClassifier(
       node_features=converter.n_node_features,
       edge_features=converter.n_edge_features,
       global_features=converter.n_graph_features,
   )

   # Train
   trainer = pyl.Trainer(max_epochs=10)
   trainer.fit(model, train_loader, val_loader)

   # Test
   trainer.test(model, test_loader)

American Football: BigDataBowl Data
------------------------------------

Load NFL tracking data:

.. code-block:: python

   from unravel.american_football import BigDataBowlDataset

   # Load BigDataBowl data
   bdb_dataset = BigDataBowlDataset(
       tracking_file_path="week1.csv",
       players_file_path="players.csv",
       plays_file_path="plays.csv",
   )

The workflow for converting to graphs and training is similar to soccer.

Soccer Analytics Models
-----------------------

Pressing Intensity
~~~~~~~~~~~~~~~~~~

Compute pressing intensity for a match segment:

.. code-block:: python

   from unravel.soccer import PressingIntensity
   import polars as pl

   model = PressingIntensity(dataset=polars_dataset)
   result = model.fit(
       start_time=pl.duration(minutes=1, seconds=53),
       end_time=pl.duration(minutes=2, seconds=32),
       period_id=1,
       method="teams",
       ball_method="max",
       speed_threshold=2.0,
   )

Formation Detection (EFPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Identify formations and positions:

.. code-block:: python

   from unravel.soccer import EFPI

   model = EFPI(dataset=polars_dataset)
   formations = model.fit(
       every="5m",  # Detect every 5 minutes
       substitutions="drop",
       change_threshold=0.1,
   )

Next Steps
----------

* Read the :doc:`concepts` guide to understand the core concepts
* Check out the :doc:`../tutorials/soccer_gnn` tutorial for an in-depth walkthrough
* Explore the :doc:`../api/soccer` API reference
* View the `example Jupyter notebooks <https://github.com/unravelsports/unravelsports/tree/main/examples>`_
