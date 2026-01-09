Soccer Graph Neural Networks
============================

This tutorial covers how to convert soccer tracking data into graphs and train Graph Neural Networks
for various prediction tasks.

The workflow consists of:

1. Loading tracking data with Kloppy
2. Converting to Polars DataFrame
3. Adding labels and graph IDs
4. Converting to graph structures
5. Training a GNN model
6. Making predictions

Interactive Notebooks
---------------------

We provide comprehensive Jupyter notebooks that walk through the entire process:

Quick Start Guides
~~~~~~~~~~~~~~~~~~

**PyTorch Geometric (Recommended)**

* `Quick Start Guide (PyTorch) <https://github.com/unravelsports/unravelsports/blob/main/examples/0_quick_start_guide_pyg.ipynb>`_

  * Basic 5-step workflow
  * Model training and prediction
  * Saving and loading models

**Spektral (Deprecated - Python 3.11 only)**

* `Quick Start Guide (Spektral) <https://github.com/unravelsports/unravelsports/blob/main/examples/0_quick_start_guide.ipynb>`_

  * Legacy TensorFlow/Keras approach
  * Not recommended for new projects

In-Depth Tutorials
~~~~~~~~~~~~~~~~~~

**PyTorch Geometric**

* `Comprehensive GNN Training (PyTorch) <https://github.com/unravelsports/unravelsports/blob/main/examples/1_kloppy_gnn_train_pyg.ipynb>`_

  * Multiple match processing
  * Custom graph features
  * Label engineering
  * Pickle file storage for large datasets
  * Advanced training patterns

**Spektral**

* `Comprehensive GNN Training (Spektral) <https://github.com/unravelsports/unravelsports/blob/main/examples/1_kloppy_gnn_train.ipynb>`_

  * Legacy approach for Python 3.11

Graph Configuration FAQ
~~~~~~~~~~~~~~~~~~~~~~~

* `Graphs FAQ <https://github.com/unravelsports/unravelsports/blob/main/examples/graphs_faq.md>`_

  * What is a graph?
  * Complete list of graph settings
  * All available node and edge features
  * Adjacency matrix representations
  * Custom feature functions

Key Concepts
------------

Data Preparation
~~~~~~~~~~~~~~~~

**Loading Data**

Use Kloppy to load tracking data from various providers:

.. code-block:: python

   from kloppy import sportec, skillcorner, tracab
   from unravel.soccer import KloppyPolarsDataset

   # Sportec (DFL Open Data)
   kloppy_dataset = sportec.load_open_tracking_data(
       only_alive=True,  # Only frames with ball in play
       limit=500  # Limit to 500 frames for testing
   )

   # SkillCorner
   kloppy_dataset = skillcorner.load(
       ...
       include_empty_frames=False  # IMPORTANT for non-Sportec data
   )

   # Convert to Polars
   polars_dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset)

**Adding Labels**

For supervised learning, add a ``label`` column:

.. code-block:: python

   from unravel.utils import add_dummy_label_column

   # Option 1: Dummy labels (for testing)
   polars_dataset.dataset = add_dummy_label_column(polars_dataset.dataset)

   # Option 2: Join real labels from your data
   import polars as pl

   # Your labels should have matching keys to join on
   labels = pl.DataFrame({
       "frame_id": [10000, 10001, 10002, ...],
       "label": [0, 1, 0, ...]
   })

   polars_dataset.dataset = polars_dataset.dataset.join(
       labels,
       on="frame_id",
       how="left"
   )

**Adding Graph IDs**

Group frames into graph samples:

.. code-block:: python

   from unravel.utils import add_graph_id_column

   # Each frame is a separate graph
   polars_dataset.dataset = add_graph_id_column(
       polars_dataset.dataset,
       by=["frame_id"]
   )

   # Or group by possession
   polars_dataset.dataset = add_graph_id_column(
       polars_dataset.dataset,
       by=["ball_owning_team_id", "period_id"]  # Changes when possession changes
   )

   # Or by sequences (e.g., 10-frame sequences)
   polars_dataset.dataset = polars_dataset.dataset.with_columns(
       (pl.col("frame_id") // 10).alias("sequence_id")
   )
   polars_dataset.dataset = add_graph_id_column(
       polars_dataset.dataset,
       by=["sequence_id"]
   )

Graph Conversion
~~~~~~~~~~~~~~~~

**Basic Configuration**

.. code-block:: python

   from unravel.soccer import SoccerGraphConverter

   converter = SoccerGraphConverter(
       dataset=polars_dataset,

       # Ball connections
       self_loop_ball=True,  # Add self-loop to ball node
       adjacency_matrix_connect_type="ball",  # How to connect ball

       # Graph structure
       adjacency_matrix_type="split_by_team",  # Team-based connections

       # Labels
       label_type="binary",  # Or "continuous", "multiclass"

       # Node values for specific roles
       defending_team_node_value=0.1,
       non_potential_receiver_node_value=0.1,

       # Training options
       random_seed=False,  # Permutation invariance during training
       pad=False,  # Padding for fixed-size graphs

       verbose=False,
   )

**Node Features**

Available node features for soccer (12 total):

* ``x_normed``: Normalized x-coordinate
* ``y_normed``: Normalized y-coordinate
* ``vx_normed``: Normalized x-velocity
* ``vy_normed``: Normalized y-velocity
* ``speed_normed``: Normalized speed magnitude
* ``ax_normed``: Normalized x-acceleration
* ``ay_normed``: Normalized y-acceleration
* ``acceleration_normed``: Normalized acceleration magnitude
* ``sin_direction``: Sine of movement direction
* ``cos_direction``: Cosine of movement direction
* ``distance_to_goal_normed``: Distance to opponent's goal
* ``angle_to_goal_normed``: Angle to opponent's goal

**Edge Features**

Available edge features (6-7 total):

* ``distance_normed``: Distance between nodes
* ``sin_angle``: Sine of angle between nodes
* ``cos_angle``: Cosine of angle between nodes
* ``relative_speed_normed``: Relative speed
* ``relative_vx_normed``: Relative x-velocity
* ``relative_vy_normed``: Relative y-velocity
* (Optionally) edge type indicators

**Adjacency Matrix Types**

* ``split_by_team``: Separate graphs for each team, connected via ball
* ``delaunay``: Spatial proximity based on Delaunay triangulation
* ``dense``: Fully connected graph
* ``dense_ap``: Dense for attacking team, proximity for defending
* ``dense_dp``: Dense for defending team, proximity for attacking

**Custom Features**

Add custom node or edge features:

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
   converter = SoccerGraphConverter(
       dataset=polars_dataset,
       node_feature_cols=[distance_from_center],
       ...
   )

Converting to Graph Format
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # PyTorch Geometric (recommended)
   graphs = converter.to_pytorch_graphs()

   # Spektral (deprecated)
   graphs = converter.to_spektral_graphs()

   # Save to pickle for reuse
   import pickle
   import gzip

   with gzip.open("graphs.pickle.gz", "wb") as f:
       pickle.dump(graphs, f)

   # Load from pickle
   with gzip.open("graphs.pickle.gz", "rb") as f:
       graphs = pickle.load(f)

Model Training
~~~~~~~~~~~~~~

**PyTorch Geometric Approach**

.. code-block:: python

   from unravel.utils import GraphDataset
   from unravel.classifiers import PyGLightningCrystalGraphClassifier
   import pytorch_lightning as pyl
   from torch_geometric.loader import DataLoader

   # Create dataset
   dataset = GraphDataset(graphs=graphs, format="pyg")

   # Split data (4:1:1 ratio for train:test:val)
   train_graphs, test_graphs, val_graphs = dataset.split_test_train_validation(4, 1, 1)

   # Create data loaders
   train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
   test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

   # Initialize model
   model = PyGLightningCrystalGraphClassifier(
       node_features=converter.n_node_features,
       edge_features=converter.n_edge_features,
       global_features=converter.n_graph_features,
       output_features=1,  # Binary classification
       learning_rate=0.001,
   )

   # Train
   trainer = pyl.Trainer(
       max_epochs=50,
       accelerator="auto",  # Use GPU if available
       devices=1,
       log_every_n_steps=10,
   )
   trainer.fit(model, train_loader, val_loader)

   # Test
   test_results = trainer.test(model, test_loader)

   # Save model
   trainer.save_checkpoint("model.ckpt")

   # Load model
   model = PyGLightningCrystalGraphClassifier.load_from_checkpoint("model.ckpt")

Making Predictions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Set converter to prediction mode (no labels needed)
   converter_pred = SoccerGraphConverter(
       dataset=new_data,
       prediction=True,  # Important!
       **same_settings_as_training
   )

   # Convert new data
   new_graphs = converter_pred.to_pytorch_graphs()
   new_dataset = GraphDataset(graphs=new_graphs, format="pyg")
   pred_loader = DataLoader(new_dataset, batch_size=32)

   # Predict
   predictions = trainer.predict(model, pred_loader)
