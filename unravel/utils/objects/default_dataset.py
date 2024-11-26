from dataclasses import dataclass


@dataclass
class DefaultDataset:
    def load(self):
        raise NotImplementedError()

    def add_dummy_labels(self):
        raise NotImplementedError()

    def add_graph_ids(self):
        raise NotImplementedError()
