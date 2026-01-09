Formation Detection (EFPI)
==========================

EFPI (Elastic Formation and Position Identification) is an algorithm for detecting team formations
and assigning tactical positions to players in soccer. This tutorial explains how to use EFPI
with the unravelsports package.

EFPI uses template matching and linear assignment to:

* **Detect formations**: Identify which formation (4-4-2, 4-3-3, etc.) a team is using
* **Assign positions**: Map each player to a tactical role (CB, CM, LW, etc.)
* **Track changes**: Monitor formation transitions throughout the match
* **Handle substitutions**: Automatically adjust for player substitutions

For the mathematical formulation and validation, see:
`Bekkers (2025): EFPI: Elastic Formation and Position Identification in Football (Soccer) <https://arxiv.org/pdf/2506.23843>`_

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

Step 2: Initialize EFPI Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an EFPI model instance:

.. code-block:: python

   from unravel.soccer import EFPI

   model = EFPI(dataset=polars_dataset)

Step 3: Detect Formations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run formation detection:

.. code-block:: python

   result = model.fit(
       formations=None,  # Use all 65 default formations
       every="5m",       # Detect every 5 minutes
       substitutions="drop",
       change_threshold=0.1,
       change_after_possession=True,
   )

Parameters Explained
--------------------

Formations
~~~~~~~~~~

Which formations to consider:

* ``formations=None``: Use all 65 default formations (recommended)
* ``formations=["442", "433", "352"]``: Only consider specific formations
* ``formations=["4231", "433"]``: Narrow search space if you know likely formations

.. code-block:: python

   # Use all formations
   result = model.fit(formations=None)

   # Only common formations
   result = model.fit(formations=["442", "433", "4231", "352", "343"])

   # Single formation (useful for validation)
   result = model.fit(formations=["442"])

Available formations include: 4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 3-4-3, 5-3-2, and many more.

Time Granularity (every)
~~~~~~~~~~~~~~~~~~~~~~~~~

How frequently to detect formations:

* ``every="frame"``: Detect for every single frame (very detailed, slow)
* ``every="5m"``: Detect every 5 minutes (good for match overview)
* ``every="1m"``: Detect every 1 minute (more granular)
* ``every="possession"``: Detect once per possession
* ``every="period"``: Detect once per period (very coarse)

.. code-block:: python

   # Frame-by-frame (most accurate but slowest)
   result = model.fit(every="frame")

   # Time-based intervals
   result = model.fit(every="1m")       # Every minute
   result = model.fit(every="30s")      # Every 30 seconds
   result = model.fit(every="5m")       # Every 5 minutes

   # Possession-based
   result = model.fit(every="possession")

   # Period-based (coarsest)
   result = model.fit(every="period")

The ``every`` parameter uses Polars' ``group_by_dynamic`` syntax. See
`Polars Documentation <https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by_dynamic.html>`_
for more options.

Substitutions
~~~~~~~~~~~~~

How to handle player substitutions:

* ``substitutions="drop"``: Remove frames where substitutions are occurring (recommended)
* ``substitutions="keep"``: Include all frames, even during substitutions
* ``substitutions="interpolate"``: Interpolate formations across substitution events

.. code-block:: python

   # Drop substitution frames (cleanest results)
   result = model.fit(substitutions="drop")

   # Keep all frames
   result = model.fit(substitutions="keep")

Change Threshold
~~~~~~~~~~~~~~~~

Minimum change required to register a new formation:

* ``change_threshold=0.1``: 10% difference required (default, balanced)
* ``change_threshold=0.0``: No threshold (very sensitive to changes)
* ``change_threshold=0.2``: 20% difference required (less sensitive)

.. code-block:: python

   # Very sensitive (may detect minor tactical adjustments)
   result = model.fit(change_threshold=0.0)

   # Balanced (default)
   result = model.fit(change_threshold=0.1)

   # Conservative (only detect major formation changes)
   result = model.fit(change_threshold=0.2)

Change After Possession
~~~~~~~~~~~~~~~~~~~~~~~~

Whether to allow formation changes mid-possession:

* ``change_after_possession=True``: Only change formation at possession boundaries (recommended)
* ``change_after_possession=False``: Allow changes at any time

.. code-block:: python

   # Only change formation when possession changes (more realistic)
   result = model.fit(change_after_possession=True)

   # Allow immediate changes (may detect temporary adjustments)
   result = model.fit(change_after_possession=False)

Output Format
-------------

The model stores results in ``model.output``, a Polars DataFrame containing:

* ``object_id``: Player ID
* ``team_id``: Team ID
* ``position``: Assigned position label (e.g., "LW", "CM", "GK")
* ``formation``: Formation name (e.g., "4-3-3")
* ``is_attacking``: Boolean indicating attacking (True) or defending (False)
* Additional columns depending on ``every`` parameter (frame_id, segment_id, etc.)

.. code-block:: python

   model.fit(every="5m")

   # Access output DataFrame
   print(model.output)

   # Filter to specific team
   home_formations = model.output.filter(pl.col("team_id") == "home")

   # Get unique formations
   formations = model.output.select(["formation", "is_attacking"]).unique()
