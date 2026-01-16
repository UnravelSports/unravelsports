Classifiers
===========

Graph Neural Network classifiers for sports analytics.

.. currentmodule:: unravel.classifiers

The classifiers module provides pre-built Graph Neural Network architectures optimized for sports
tracking data. These models can be used with both PyTorch Geometric and Spektral (deprecated).

PyTorch Geometric
-----------------

.. autoclass:: PyGCrystalGraphClassifier
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PyGLightningCrystalGraphClassifier
   :show-inheritance:
   :no-index:

Spektral
--------

.. autoclass:: CrystalGraphClassifier
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

PyTorch Geometric
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from unravel.classifiers import PyGLightningCrystalGraphClassifier
   import pytorch_lightning as pyl
   from torch_geometric.loader import DataLoader

   # Initialize model
   model = PyGLightningCrystalGraphClassifier()

   # Train
   trainer = pyl.Trainer(max_epochs=50)
   trainer.fit(model, train_loader, val_loader)

   # Test
   trainer.test(model, test_loader)

Spektral
~~~~~~~~

.. code-block:: python

   from unravel.classifiers import CrystalGraphClassifier

   from tensorflow.keras.metrics import AUC, BinaryAccuracy
   from tensorflow.keras.losses import BinaryCrossentropy
   from tensorflow.keras.optimizers import Adam
   from tensorflow.keras.callbacks import EarlyStopping

   model = CrystalGraphClassifier()

   model.compile(
      loss=BinaryCrossentropy(), optimizer=Adam(), metrics=[AUC(), BinaryAccuracy()]
   )

   model.fit(
      loader_tr.load(),
      steps_per_epoch=loader_tr.steps_per_epoch,
      epochs=5,
      use_multiprocessing=True,
      validation_data=loader_va.load(),
      callbacks=[EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)],
   )

   from tensorflow.keras.models import load_model

   model_path = "models/my-first-graph-classifier"
   model.save(model_path)
   loaded_model = load_model(model_path)

   loader_te = DisjointLoader(test, epochs=1, shuffle=False, batch_size=batch_size)
   results = model.evaluate(loader_te.load())