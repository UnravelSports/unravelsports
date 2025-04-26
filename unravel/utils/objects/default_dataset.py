from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class DefaultDataset:
    _graph_id_column: str = field(default="graph_id", repr=False)
    _label_column: str = field(default="label", repr=False)

    # Subclasses should set these in __post_init__ or in load()
    data: Optional[Any] = field(init=False, default=None, repr=False)
    settings: Any = field(init=False, repr=False)

    def load(self):
        raise NotImplementedError()

    def add_dummy_labels(self):
        raise NotImplementedError()

    def add_graph_ids(self):
        raise NotImplementedError()
    
    def get_dataframe(self) -> Optional[Any]:
        """
        Return the loaded DataFrame (or None if load() hasn't been called yet).
        Recommended for downstream users to fetch raw data.
        """
        return getattr(self, "data", None)

    def get_settings(self) -> Any:
        """
        Return this dataset's settings object.
        Allows uniform access across all DefaultDataset subclasses.
        """
        return getattr(self, "settings")
