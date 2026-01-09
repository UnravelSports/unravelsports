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
   :members:
   :undoc-members:
   :show-inheritance:

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
   model = PyGLightningCrystalGraphClassifier(
       node_features=12,
       edge_features=6,
       global_features=0,
       output_features=1,
       learning_rate=0.001,
   )

   # Train
   trainer = pyl.Trainer(max_epochs=50)
   trainer.fit(model, train_loader, val_loader)

   # Test
   trainer.test(model, test_loader)

   # Predict
   predictions = trainer.predict(model, pred_loader)

Spektral
~~~~~~~~

.. code-block:: python

   from unravel.classifiers import CrystalGraphClassifier

   # Initialize model
   model = CrystalGraphClassifier(
       node_features=12,
       edge_features=6,
       output_features=1,
   )

   # Compile
   model.compile(
       optimizer='adam',
       loss='binary_crossentropy',
       metrics=['accuracy']
   )

   # Train
   model.fit(x=train_data, y=train_labels, epochs=50, validation_data=(val_data, val_labels))
