Core Concepts
=============

This page explains the key concepts and terminology used in unravelsports.

Data Flow
---------

The typical data flow in unravelsports follows these steps:

1. **Raw Tracking Data** → Loaded via Kloppy (soccer) or direct CSV (American Football)
2. **Polars DataFrame** → Fast, efficient data representation
3. **Graph Structures** → Convert to graphs for GNN training
4. **Model Training** → Train with PyTorch Geometric or Spektral
5. **Predictions/Analytics** → Apply models or compute metrics

Tracking Data
-------------

What is Tracking Data?
~~~~~~~~~~~~~~~~~~~~~~

Tracking data captures the position of players and the ball at high frequency (typically 10-25 Hz).
Each frame includes:

* **x, y coordinates**: Position on the pitch/field
* **Velocities**: Speed in x and y directions
* **Ball state**: Whether the ball is in play, out of bounds, etc.
* **Metadata**: Team IDs, player IDs, timestamps

Supported Data Providers
~~~~~~~~~~~~~~~~~~~~~~~~~

Soccer (via Kloppy):

* Sportec (DFL Open Data)
* SkillCorner
* Tracab (ChyronHego)
* Second Spectrum
* StatsPerform
* Metrica Sports
* PFF / GradientSports
* HawkEye
* Signality

American Football:

* NFL Big Data Bowl CSV files

Polars DataFrames
-----------------

Why Polars?
~~~~~~~~~~~

Polars is a blazingly fast DataFrame library written in Rust with a Python API. Benefits:

* **Performance**: 10-100x faster than pandas for many operations
* **Memory efficiency**: Lower memory footprint
* **Lazy evaluation**: Build query plans before execution
* **Modern API**: Clean, consistent interface

DataFrame Structure
~~~~~~~~~~~~~~~~~~~

After conversion, the DataFrame contains:

* ``period_id``: Match period (1, 2, etc.)
* ``timestamp``: Time within the period
* ``frame_id``: Unique frame identifier
* ``ball_state``: alive, dead, out, etc.
* ``id``: Player/ball ID
* ``x, y, z``: Coordinates
* ``team_id``: Team identifier
* ``position_name``: Player position (GK, CB, etc.)
* ``vx, vy, vz``: Velocity components
* ``ax, ay, az``: Acceleration components
* ``ball_owning_team_id``: Team in possession
* ``is_ball_carrier``: Whether player has the ball

Graph Neural Networks
---------------------

What are Graphs?
~~~~~~~~~~~~~~~~

In the context of sports tracking data, a graph represents:

* **Nodes**: Players and the ball
* **Edges**: Relationships between players (teammates, opponents, proximity to ball)
* **Node Features**: Player attributes (position, velocity, acceleration, etc.)
* **Edge Features**: Relationship attributes (distance, angle, relative velocity)
* **Global Features**: Game-level information (score, time, etc.)

Why Use GNNs for Sports Data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph Neural Networks are ideal for sports analytics because:

1. **Permutation Invariance**: Player order doesn't matter
2. **Relational Reasoning**: Capture interactions between players
3. **Variable Size**: Handle different numbers of players on the field
4. **Spatial Structure**: Naturally model the spatial nature of sports

Graph Conversion Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Key parameters when converting to graphs:

* **adjacency_matrix_type**: How to connect nodes

  * ``split_by_team``: Separate connections for each team
  * ``delaunay``: Based on spatial proximity (Delaunay triangulation)
  * ``dense``: Fully connected graph

* **adjacency_matrix_connect_type**: How to connect to the ball

  * ``ball``: Connect all players to the ball
  * ``ball_carrier``: Only connect ball carrier to ball
  * ``no_connection``: No ball connections

* **Node features**: What information to include for each node
* **Edge features**: What information to include for each edge

See the :doc:`../tutorials/soccer_gnn` tutorial and
`Graph FAQ <https://github.com/unravelsports/unravelsports/blob/main/examples/graphs_faq.md>`_
for more details.

Labels and Graph IDs
--------------------

Labels
~~~~~~

For supervised learning, you need labels for each graph:

.. code-block:: python

   from unravel.utils import add_dummy_label_column

   # Add random binary labels (for demonstration)
   dataset.dataset = add_dummy_label_column(dataset.dataset)

   # Or join real labels from your own data
   # dataset.dataset = dataset.dataset.join(your_labels, on="some_key")

Graph IDs
~~~~~~~~~

Graph IDs group frames that belong to the same "sample":

.. code-block:: python

   from unravel.utils import add_graph_id_column

   # Each frame is a separate graph
   dataset.dataset = add_graph_id_column(dataset.dataset, by=["frame_id"])

   # Or group by possession
   dataset.dataset = add_graph_id_column(dataset.dataset, by=["possession_id"])

**Important**: Always split data by graph_id to avoid data leakage!

Soccer Analytics Models
-----------------------

Pressing Intensity
~~~~~~~~~~~~~~~~~~

A metric quantifying defensive pressure on ball carriers. Based on:

* Defender positions relative to ball carrier
* Defender velocities
* Spatial coverage

See `Bekkers (2024) <https://arxiv.org/pdf/2501.04712>`_ for the mathematical formulation.

EFPI (Elastic Formation and Position Identification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A template matching algorithm to:

1. Detect team formations (4-4-2, 4-3-3, etc.)
2. Assign tactical positions to players
3. Handle substitutions and formation changes

Uses linear assignment to match player positions to formation templates.

See `Bekkers (2025) <https://arxiv.org/pdf/2506.23843>`_ for details.
