Pressing Intensity
==================

Pressing Intensity is a metric that quantifies the defensive pressure applied to ball carriers
in soccer. This tutorial explains how to compute and visualize pressing intensity using the
unravelsports package.

The Pressing Intensity metric measures:

* **Spatial pressure**: How closely defenders surround the ball carrier
* **Velocity pressure**: How fast defenders are moving toward the ball carrier
* **Coverage**: How well defenders cover potential passing lanes

For the mathematical formulation and validation, see:
`Bekkers (2024): Pressing Intensity: An Intuitive Measure for Pressing in Soccer <https://arxiv.org/pdf/2501.04712>`_

Interactive Notebook
--------------------

A comprehensive Jupyter notebook demonstrates the full workflow, including video generation:

* `Pressing Intensity Tutorial <https://github.com/unravelsports/unravelsports/blob/main/examples/pressing_intensity.ipynb>`_

  * Loading tracking data
  * Computing pressing intensity for match segments
  * Creating MP4 visualizations with matplotlib and mplsoccer
  * Example: 1. FC Köln vs. FC Bayern München (May 27th 2023)

Basic Usage
-----------

Step 1: Load Tracking Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load your soccer tracking data:

.. code-block:: python

   from kloppy import sportec
   from unravel.soccer import KloppyPolarsDataset

   # Load tracking data
   kloppy_dataset = sportec.load_open_tracking_data(
       only_alive=True
   )

   # Convert to Polars format
   polars_dataset = KloppyPolarsDataset(
       kloppy_dataset=kloppy_dataset
   )

Step 2: Initialize Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a PressingIntensity model instance:

.. code-block:: python

   from unravel.soccer import PressingIntensity

   model = PressingIntensity(dataset=polars_dataset)

Step 3: Compute Pressing Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute pressing intensity for a specific time window:

.. code-block:: python

   import polars as pl

   model.fit(
       start_time=pl.duration(minutes=1, seconds=53),
       end_time=pl.duration(minutes=2, seconds=32),
       period_id=1,
       method="teams",
       ball_method="max",
       orient="home_away",
       speed_threshold=2.0,
   )

   # Access results
   print(model.output)

Parameters Explained
--------------------

Time Window
~~~~~~~~~~~

* ``start_time``: Start of the analysis window (Polars duration)
* ``end_time``: End of the analysis window (Polars duration)
* ``period_id``: Which period to analyze (1 for first half, 2 for second half)

.. code-block:: python

   import polars as pl

   # First minute of the match
   start_time = pl.duration(minutes=0, seconds=0)
   end_time = pl.duration(minutes=1, seconds=0)

   # Specific moment (e.g., 15:30 - 16:00)
   start_time = pl.duration(minutes=15, seconds=30)
   end_time = pl.duration(minutes=16, seconds=0)

   # Or analyze entire period
   start_time = None  # Start from beginning
   end_time = None    # Until end of period

Method
~~~~~~

Controls the matrix structure:

* ``method="teams"``: Creates 11×11 matrix (ball-owning team × non-owning team)
* ``method="full"``: Creates 22×22 matrix (all players × all players)

Ball Method
~~~~~~~~~~~

How to handle the ball in the pressing intensity matrix:

* ``ball_method="max"``: Merge ball with ball carrier using max(ball_tti, carrier_tti) - keeps 11×11 matrix
* ``ball_method="include"``: Add ball as separate node (creates 11×12 or 22×23 matrix)
* ``ball_method="exclude"``: Ignore ball entirely

Orientation
~~~~~~~~~~~

Matrix orientation perspective:

* ``orient="ball_owning"``: Rows = ball-owning team, Cols = non-owning team
* ``orient="pressing"``: Rows = non-owning team, Cols = ball-owning team (transpose)
* ``orient="home_away"``: Rows = home team, Cols = away team
* ``orient="away_home"``: Rows = away team, Cols = home team

Speed Threshold
~~~~~~~~~~~~~~~

Minimum speed (m/s) for a player to be considered actively pressing:

* ``speed_threshold=2.0``: Players moving faster than 2 m/s (filters out passive coverage)
* ``speed_threshold=None``: All players included regardless of speed (default)

Output Format
-------------

The model stores results in ``model.output``, a Polars DataFrame with one row per frame containing:

* ``frame_id``, ``period_id``, ``timestamp``: Frame identifiers
* ``time_to_intercept``: List[List[float]] - TTI matrix (rows × columns)
* ``probability_to_intercept``: List[List[float]] - PTI matrix (rows × columns)
* ``columns``: List[str] - Object IDs for column players
* ``rows``: List[str] - Object IDs for row players

.. code-block:: python

   model.fit(
       start_time=pl.duration(minutes=0),
       end_time=pl.duration(minutes=1),
       period_id=1,
       method="teams"
   )

   # Access output DataFrame
   print(model.output)

   # Extract matrices for a specific frame
   frame_data = model.output.filter(pl.col("frame_id") == 1000)
   tti_matrix = np.array(frame_data["time_to_intercept"][0])
   pti_matrix = np.array(frame_data["probability_to_intercept"][0])

Visualization
-------------

For complete visualization examples with heatmaps and video generation, see the
`Pressing Intensity Jupyter Notebook <https://github.com/unravelsports/unravelsports/blob/main/examples/pressing_intensity.ipynb>`_.

The notebook demonstrates:

* Creating animated heatmaps of pressing intensity matrices
* Overlaying pressing intensity on pitch visualizations
* Generating MP4 videos with matplotlib and mplsoccer
