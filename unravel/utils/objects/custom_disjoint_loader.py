from spektral.data import DisjointLoader
import tensorflow as tf

version = tf.__version__.split(".")
major, minor = int(version[0]), int(version[1])
tf_loader_available = major >= 2 and minor >= 4


class CustomDisjointLoader(DisjointLoader):
    def __init__(
        self, dataset, node_level=False, batch_size=1, epochs=None, shuffle=True
    ):
        self.node_level = node_level
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def load(self):
        if not tf_loader_available:
            raise RuntimeError(
                "Calling DisjointLoader.load() requires TensorFlow 2.4 or greater."
            )
        dataset = tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.tf_signature()
        )
        dataset = dataset.shuffle(buffer_size=1000)
        return dataset.repeat()
