from spektral.layers import GlobalAvgPool, CrystalConv
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model


class CrystalGraphClassifier(Model):
    """
    Default Graph Classifier with CrystalConvolution layers as presented in Sahasrabudhe & Bekkers (2023)
    """

    def __init__(
        self,
        n_layers: int = 3,
        channels: int = 128,
        drop_out: float = 0.5,
        n_out: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_layers = n_layers
        self.channels = channels
        self.drop_out = drop_out
        self.n_out = n_out

        self.conv1 = CrystalConv()
        self.convs = [CrystalConv() for _ in range(1, self.n_layers)]
        self.pool = GlobalAvgPool()
        self.dense1 = Dense(self.channels, activation="relu")
        self.dropout = Dropout(self.drop_out)
        self.dense2 = Dense(self.channels, activation="relu")
        self.dense3 = Dense(self.n_out, activation="sigmoid")

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.conv1([x, a, e])
        for conv in self.convs:
            x = conv([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return self.dense3(x)
