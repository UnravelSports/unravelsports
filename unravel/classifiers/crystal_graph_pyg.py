try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import CGConv, global_mean_pool

    _HAS_TORCH_GEOMETRIC = True
    _BASE_CLASS = nn.Module
except ImportError:
    _HAS_TORCH_GEOMETRIC = False
    _BASE_CLASS = object

try:
    import pytorch_lightning as pyl
    from torchmetrics import AUROC, Accuracy

    _HAS_PYTORCH_LIGHTNING = True
    _PYL_BASE_CLASS = pyl.LightningModule
except ImportError:
    _HAS_PYTORCH_LIGHTNING = False
    _PYL_BASE_CLASS = object


class PyGCrystalGraphClassifier(_BASE_CLASS):
    """Graph Neural Network classifier using Crystal Graph Convolutional layers.

    This classifier uses CGConv (Crystal Graph Convolutional) layers that incorporate
    edge features in the message passing mechanism. It's designed for graph-level
    classification tasks on sports tracking data.

    The architecture consists of:
    - Input projection layers (for variable-sized node and edge features)
    - Multiple CGConv layers for graph convolution
    - Global mean pooling to aggregate node features
    - Dense layers with dropout for classification

    Args:
        n_layers: Number of CGConv layers. Defaults to 3.
        channels: Hidden dimension size for all layers. Defaults to 128.
        drop_out: Dropout probability for regularization. Defaults to 0.5.
        n_out: Number of output features (1 for binary classification). Defaults to 1.
        **kwargs: Additional keyword arguments (currently unused).

    Raises:
        ImportError: If PyTorch Geometric is not installed.

    Example:
        >>> from unravel.classifiers import PyGCrystalGraphClassifier
        >>> model = PyGCrystalGraphClassifier(
        ...     n_layers=3,
        ...     channels=128,
        ...     drop_out=0.5,
        ...     n_out=1
        ... )
        >>> # Forward pass
        >>> output = model(x, edge_index, edge_attr, batch)

    Note:
        This is a pure PyTorch model. For automatic training loops with logging,
        use :class:`PyGLightningCrystalGraphClassifier` instead.
    """

    def __init__(
        self,
        n_layers: int = 3,
        channels: int = 128,
        drop_out: float = 0.5,
        n_out: int = 1,
        **kwargs
    ):
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "PyTorch Geometric is required for PyGCrystalGraphClassifier. "
                "Install it using: pip install torch torch-geometric pytorch-lightning torchmetrics"
            )
        super().__init__()

        self.n_layers = n_layers
        self.channels = channels
        self.drop_out = drop_out
        self.n_out = n_out

        # Project variable node features to fixed size
        self.input_projection = nn.LazyLinear(channels)

        # Project variable edge features to fixed size
        self.edge_projection = nn.LazyLinear(channels)

        # CGConv layers with edge features
        # dim should be the edge feature dimension AFTER projection
        self.convs = nn.ModuleList(
            [
                CGConv(
                    channels, dim=channels
                )  # Edge features have 'channels' dimensions after projection
                for _ in range(self.n_layers)
            ]
        )

        # Dense layers
        self.dense1 = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(drop_out)
        self.dense2 = nn.Linear(channels, channels)
        self.dense3 = nn.Linear(channels, n_out)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """Forward pass through the graph neural network.

        Processes a batch of graphs through the network layers and returns
        graph-level predictions.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels].
                Can have variable in_channels as they are projected to fixed size.
            edge_index (torch.LongTensor): Graph connectivity in COO format with shape
                [2, num_edges].
            edge_attr (torch.Tensor, optional): Edge feature matrix with shape
                [num_edges, edge_features]. Can have variable edge_features as they
                are projected to fixed size. Defaults to None.
            batch (torch.LongTensor, optional): Batch vector with shape [num_nodes]
                which assigns each node to a specific example in the batch.
                Defaults to None (single graph).

        Returns:
            torch.Tensor: Graph-level predictions with shape [batch_size, n_out].
                Values are in range [0, 1] after sigmoid activation.

        Example:
            >>> # Single graph
            >>> output = model(x, edge_index, edge_attr, batch=None)
            >>> # Batch of graphs
            >>> output = model(x, edge_index, edge_attr, batch)
        """
        # Project node features to fixed size
        x = self.input_projection(x)

        # Project edge features to fixed size (if they exist)
        if edge_attr is not None:
            edge_attr = self.edge_projection(edge_attr)

        # Apply CGConv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Dense layers with dropout
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.dense3(x))

        return x


class PyGLightningCrystalGraphClassifier(_PYL_BASE_CLASS):
    """PyTorch Lightning wrapper for Crystal Graph Classifier with training loop.

    This class wraps :class:`PyGCrystalGraphClassifier` with PyTorch Lightning
    functionality, providing automatic training loops, logging, checkpointing,
    and metrics tracking for binary classification tasks.

    The model includes:
    - Automatic training/validation/test loops
    - AUROC and accuracy metric tracking
    - Learning rate scheduling with ReduceLROnPlateau
    - Automatic checkpointing and logging
    - Easy prediction interface

    Args:
        n_layers (int, optional): Number of CGConv layers. Defaults to 3.
        channels (int, optional): Hidden dimension size. Defaults to 128.
        drop_out (float, optional): Dropout probability. Defaults to 0.5.
        n_out (int, optional): Number of output features. Defaults to 1.
        lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
        weight_decay (float, optional): L2 penalty coefficient. Defaults to 0.0.

    Raises:
        ImportError: If PyTorch Lightning or torchmetrics is not installed.

    Attributes:
        model (PyGCrystalGraphClassifier): The underlying GNN model.
        criterion (torch.nn.BCELoss): Binary cross-entropy loss function.
        train_auc (AUROC): Training AUROC metric.
        train_acc (Accuracy): Training accuracy metric.
        val_auc (AUROC): Validation AUROC metric.
        val_acc (Accuracy): Validation accuracy metric.
        test_auc (AUROC): Test AUROC metric.
        test_acc (Accuracy): Test accuracy metric.

    Example:
        >>> from unravel.classifiers import PyGLightningCrystalGraphClassifier
        >>> import pytorch_lightning as pyl
        >>> from torch_geometric.loader import DataLoader
        >>>
        >>> # Initialize model
        >>> model = PyGLightningCrystalGraphClassifier(
        ...     n_layers=3,
        ...     channels=128,
        ...     lr=0.001
        ... )
        >>>
        >>> # Train
        >>> trainer = pyl.Trainer(max_epochs=50, accelerator="auto")
        >>> trainer.fit(model, train_loader, val_loader)
        >>>
        >>> # Test
        >>> trainer.test(model, test_loader)
        >>>
        >>> # Predict
        >>> predictions = trainer.predict(model, pred_loader)
        >>>
        >>> # Save/load checkpoint
        >>> trainer.save_checkpoint("model.ckpt")
        >>> model = PyGLightningCrystalGraphClassifier.load_from_checkpoint("model.ckpt")

    Note:
        This model uses binary cross-entropy loss and is designed for binary
        classification tasks. For multi-class or regression tasks, you may need
        to modify the loss function and output activation.
    """

    def __init__(
        self,
        n_layers=3,
        channels=128,
        drop_out=0.5,
        n_out=1,
        lr=0.001,
        weight_decay=0.0,
    ):
        if not _HAS_PYTORCH_LIGHTNING:
            raise ImportError(
                "PyTorch Lightning is required for PyGLightningCrystalGraphClassifier. "
                "Install it using: pip install pytorch-lightning torchmetrics"
            )
        super().__init__()
        self.save_hyperparameters()

        self.model = PyGCrystalGraphClassifier(
            n_layers=n_layers, channels=channels, drop_out=drop_out, n_out=n_out
        )
        self.criterion = torch.nn.BCELoss()

        # Training metrics
        self.train_auc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")

        # Validation metrics
        self.val_auc = AUROC(task="binary")
        self.val_acc = Accuracy(task="binary")

        # Test metrics (ADD THESE!)
        self.test_auc = AUROC(task="binary")
        self.test_acc = Accuracy(task="binary")

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.LongTensor): Edge indices.
            edge_attr (torch.Tensor): Edge features.
            batch (torch.LongTensor): Batch vector.

        Returns:
            torch.Tensor: Predictions with shape [batch_size].
        """
        return self.model(x, edge_index, edge_attr, batch).squeeze(-1)

    def training_step(self, batch, batch_idx):
        """Training step executed for each batch.

        Args:
            batch: Batch of graph data from DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss for this batch.
        """
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.criterion(out, batch.y.float())

        self.train_auc(out, batch.y.int())
        self.train_acc(out, batch.y.int())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step executed for each batch.

        Args:
            batch: Batch of graph data from DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Validation loss for this batch.
        """
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.criterion(out, batch.y.float())

        self.val_auc(out, batch.y.int())
        self.val_acc(out, batch.y.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step for model evaluation.

        Computes test loss and metrics (AUROC and accuracy) for the given batch.

        Args:
            batch: Batch of graph data from DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Test loss for this batch.
        """
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.criterion(out, batch.y.float())

        # Use the class-level test metrics
        self.test_auc(out, batch.y.int())
        self.test_acc(out, batch.y.int())

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference.

        Returns predicted probabilities for the given batch without computing loss.

        Args:
            batch: Batch of graph data from DataLoader.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Predicted probabilities with shape [batch_size].
                Values are in range [0, 1].

        Example:
            >>> predictions = trainer.predict(model, pred_loader)
            >>> # predictions is a list of tensors, one per batch
            >>> all_preds = torch.cat(predictions)
        """
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return out

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Uses Adam optimizer with learning rate scheduling via ReduceLROnPlateau.
        The learning rate is reduced by a factor of 0.5 when validation loss
        plateaus for 3 epochs.

        Returns:
            dict: Dictionary containing:
                - 'optimizer': Adam optimizer instance
                - 'lr_scheduler': Dict with scheduler and monitoring configuration

        Note:
            The learning rate scheduler monitors 'val_loss' and reduces the
            learning rate when validation loss stops improving.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
