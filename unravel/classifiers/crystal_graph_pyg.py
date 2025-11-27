try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import CGConv, global_mean_pool

    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False

try:
    import pytorch_lightning as pyl
    from torchmetrics import AUROC, Accuracy

    _HAS_PYTORCH_LIGHTNING = True
except ImportError:
    _HAS_PYTORCH_LIGHTNING = False


class PyGCrystalGraphClassifier(nn.Module):
    """
    Graph Classifier with CGConv using edge features.
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
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            batch: Batch vector [num_nodes]

        Returns:
            out: Graph-level predictions [batch_size, n_out]
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


class PyGLightningCrystalGraphClassifier(pyl.LightningModule):
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
        return self.model(x, edge_index, edge_attr, batch).squeeze(-1)

    def training_step(self, batch, batch_idx):
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
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = self.criterion(out, batch.y.float())

        self.val_auc(out, batch.y.int())
        self.val_acc(out, batch.y.int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step for evaluation"""
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
        """Prediction step - returns probabilities"""
        out = self(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return out

    def configure_optimizers(self):
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
