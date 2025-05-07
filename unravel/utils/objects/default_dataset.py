from dataclasses import dataclass, field


@dataclass
class DefaultDataset:
    _graph_id_column: str = field(default="graph_id", repr=False)
    _label_column: str = field(default="label", repr=False)

    def load(self):
        raise NotImplementedError()

    def add_dummy_labels(self):
        raise NotImplementedError()

    def add_graph_ids(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()
