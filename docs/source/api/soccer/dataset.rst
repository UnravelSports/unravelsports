Dataset
=======

Loading and converting soccer tracking data.

.. currentmodule:: unravel.soccer

.. autoclass:: KloppyPolarsDataset
   :members:
   :undoc-members:
   :show-inheritance:

.. code-block:: python

   from kloppy import sportec
   from unravel.soccer import KloppyPolarsDataset

   # Load tracking data
   kloppy_dataset = sportec.load_open_tracking_data(only_alive=True)

   # Convert to Polars
   polars_dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset)

   # Access the DataFrame
   df = polars_dataset.dataset
   print(df.head())
