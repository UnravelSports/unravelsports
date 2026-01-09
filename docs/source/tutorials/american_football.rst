American Football (NFL)
=======================

This tutorial covers how to work with NFL tracking data from the Big Data Bowl using the
unravelsports package.

The unravelsports package supports NFL tracking data from the Big Data Bowl competitions,
allowing you to:

* Load and process NFL tracking data
* Convert plays to graph structures
* Train Graph Neural Networks for play prediction
* Analyze player movements and formations

Interactive Notebook
--------------------

A comprehensive Jupyter notebook walks through the entire process:

* `Big Data Bowl Guide <https://github.com/unravelsports/unravelsports/blob/main/examples/2_big_data_bowl_guide.ipynb>`_

  * Loading Big Data Bowl CSV files
  * Converting to graphs
  * Training GNN models
  * Making predictions

Data Format
-----------

Big Data Bowl Data
~~~~~~~~~~~~~~~~~~

The Big Data Bowl provides three main CSV files:

1. **tracking_week*.csv**: Player and ball tracking data

   * ``gameId``: Unique game identifier
   * ``playId``: Unique play identifier
   * ``nflId``: Player identifier
   * ``frameId``: Frame number
   * ``x, y``: Position coordinates
   * ``s``: Speed
   * ``a``: Acceleration
   * ``dis``: Distance traveled
   * ``o``: Orientation angle
   * ``dir``: Direction of travel

2. **players.csv**: Player information

   * ``nflId``: Player identifier
   * ``height``: Player height
   * ``weight``: Player weight
   * ``position``: Player position (QB, RB, WR, etc.)

3. **plays.csv**: Play-level information

   * ``gameId``, ``playId``: Identifiers
   * ``quarter``: Quarter number
   * ``down``, ``yardsToGo``: Down and distance
   * ``possessionTeam``: Team with possession
   * ``offenseFormation``: Formation name
   * ``defendersInTheBox``: Number of box defenders
   * (and many more columns)

Basic Usage
-----------

Step 1: Load Data
~~~~~~~~~~~~~~~~~

Load the Big Data Bowl CSV files:

.. code-block:: python

   from unravel.american_football import BigDataBowlDataset

   # Load data
   bdb_dataset = BigDataBowlDataset(
       tracking_file_path="tracking_week_1.csv",
       players_file_path="players.csv",
       plays_file_path="plays.csv",
   )

   # View the data
   print(bdb_dataset.dataset.head())

The resulting Polars DataFrame includes all tracking data merged with player and play information.

Step 2: Add Labels and Graph IDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For supervised learning, add labels and graph IDs:

.. code-block:: python

   from unravel.utils import add_dummy_label_column, add_graph_id_column

   # Add labels (use your own labels for real tasks)
   bdb_dataset.dataset = add_dummy_label_column(bdb_dataset.dataset)

   # Create graph ID for each play
   bdb_dataset.dataset = add_graph_id_column(
       bdb_dataset.dataset,
       by=["gameId", "playId"]
   )

Step 3: Convert to Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert tracking data to graph structures:

.. code-block:: python

   from unravel.american_football import AmericanFootballGraphConverter

   converter = AmericanFootballGraphConverter(
       dataset=bdb_dataset,
       self_loop_ball=True,
       adjacency_matrix_connect_type="ball",
       adjacency_matrix_type="split_by_team",
       label_type="binary",
   )

   # Convert to PyTorch Geometric graphs
   graphs = converter.to_pytorch_graphs()

Step 4: Train a Model
~~~~~~~~~~~~~~~~~~~~~~

Train a Graph Neural Network:

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

   # Initialize and train model
   model = PyGLightningCrystalGraphClassifier(
       node_features=converter.n_node_features,
       edge_features=converter.n_edge_features,
       global_features=converter.n_graph_features,
   )

   trainer = pyl.Trainer(max_epochs=10)
   trainer.fit(model, train_loader, val_loader)
   trainer.test(model, test_loader)

Data Availability
-----------------

Big Data Bowl data is released annually for Kaggle competitions:

* `Big Data Bowl Homepage <https://www.kaggle.com/c/nfl-big-data-bowl-2025>`_
* Previous years' data available for download
* Includes selected weeks from NFL season
* Requires Kaggle account (free)
