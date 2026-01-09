Models
======

Soccer-specific analytical models.

.. currentmodule:: unravel.soccer

.. autoclass:: PressingIntensity
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: EFPI
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.soccer import PressingIntensity
   import polars as pl

   model = PressingIntensity(dataset=polars_dataset)
   result = model.fit(
       start_time=pl.duration(minutes=1, seconds=53),
       end_time=pl.duration(minutes=2, seconds=32),
       period_id=1,
       method="teams",
   )

Formation Detection (EFPI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from unravel.soccer import EFPI

   model = EFPI(dataset=polars_dataset)
   formations = model.fit(
       every="5m",
       substitutions="drop",
   )
