Dataset
=======

Loading NFL tracking data from Big Data Bowl.

.. currentmodule:: unravel.american_football

.. autoclass:: BigDataBowlDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from unravel.american_football import BigDataBowlDataset

   bdb_dataset = BigDataBowlDataset(
       tracking_file_path="tracking_week_1.csv",
       players_file_path="players.csv",
       plays_file_path="plays.csv",
   )

   # Access the DataFrame
   df = bdb_dataset.dataset
   print(df.head())
