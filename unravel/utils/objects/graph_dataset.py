import logging
import sys
from typing import List, Tuple, Union, Optional, Literal

import numpy as np

import gzip
import pickle
from pathlib import Path

import warnings

from collections.abc import Sequence

from unravel.utils.exceptions import NoGraphIdsWarning, SpektralDependencyError


def load_pickle_gz(file_path):
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


class _GraphDatasetMixin:
    """
    Base mixin for graph dataset functionality.
    Framework-agnostic implementation that works with both Spektral and PyTorch Geometric.
    """

    def __init__(self, **kwargs):
        """
        Constructor to load parameters.

        Args:
            pickle_folder: Path to folder containing .pickle.gz files
            pickle_file: Path to single .pickle.gz file
            graphs: List of graph objects (Spektral Graph, PyG Data, or dicts)
            format: Optional explicit format specification ('spektral' or 'pyg')
            sample_rate: Sampling rate (1.0 = use all data)
        """
        self._kwargs = kwargs
        self._explicit_format = kwargs.get("format", None)

        sample_rate = kwargs.get("sample_rate", 1.0)
        self.sample = 1.0 / sample_rate

        if kwargs.get("pickle_folder", None):
            pickle_folder = Path(kwargs["pickle_folder"])
            self.graphs = None
            # Loop over all .pickle.gz files in the folder
            for pickle_file in pickle_folder.glob("*.pickle.gz"):
                data = load_pickle_gz(pickle_file)
                if not self.graphs:
                    self.graphs = self.__convert(data)
                else:
                    self.add(data)

        elif kwargs.get("pickle_file", None):
            pickle_file = Path(kwargs["pickle_file"])
            self.graphs = None
            data = load_pickle_gz(pickle_file)

            if not self.graphs:
                self.graphs = self.__convert(data)
            else:
                self.add(data)

        elif kwargs.get("graphs", None):
            if not isinstance(kwargs["graphs"], list):
                raise NotImplementedError("""data should be of type list""")

            self.graphs = self.__convert(kwargs["graphs"])
        else:
            raise NotImplementedError(
                "Please provide either 'pickle_folder', 'pickle_file' or 'graphs' as parameter to GraphDataset"
            )

        # Only call super().__init__ if there's a parent class that needs it
        # For PyGGraphDataset, Sequence doesn't take kwargs
        # For SpektralGraphDataset, Dataset does take kwargs
        try:
            super().__init__(**kwargs)
        except TypeError:
            # If super().__init__() doesn't accept kwargs (like Sequence), call it without args
            super().__init__()

    def __convert(self, data):
        """
        Convert incoming data to correct format.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement __convert()")

    def read(self):
        """
        Overriding the read function - to return a list of Graph objects.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement read()")

    def add(self, other, verbose: bool = False):
        """Add more graphs to the dataset"""
        other = self.__convert(other)

        if verbose:
            logging.info(f"Adding {len(other)} graphs to GraphDataset...")

        self.graphs = self.graphs + other

    def dimensions(self) -> Tuple[int, int, int, int, int]:
        """
        N = Max number of nodes
        F = Dimensions of Node Features
        S = Dimensions of Edge Features
        n_out = Dimension of the target
        n = Number of samples in dataset
        """
        raise NotImplementedError("Subclasses must implement dimensions()")

    def split_test_train(
        self,
        split_train: float,
        split_test: float,
        by_graph_id: bool = False,
        random_seed: Union[bool, int] = False,
        train_label_ratio: Optional[float] = None,
        test_label_ratio: Optional[float] = None,
    ):
        return self.split_test_train_validation(
            split_train=split_train,
            split_test=split_test,
            split_validation=0.0,
            by_graph_id=by_graph_id,
            random_seed=random_seed,
            train_label_ratio=train_label_ratio,
            test_label_ratio=test_label_ratio,
        )

    def split_test_train_validation(
        self,
        split_train: float,
        split_test: float,
        split_validation: float,
        by_graph_id: bool = False,
        random_seed: int = None,
        train_label_ratio: Optional[float] = None,
        test_label_ratio: Optional[float] = None,
        val_label_ratio: Optional[float] = None,
    ):
        """
        Split dataset into train, test, and validation sets with optional label balancing.
        """
        total = split_train + split_test + split_validation

        train_pct = split_train / total
        test_pct = split_test / total
        validation_pct = split_validation / total

        if by_graph_id and (
            (validation_pct > train_pct)
            or (test_pct > train_pct)
            or (validation_pct > test_pct)
        ):
            raise NotImplementedError(
                "Make sure split_train > split_test >= split_validation, other behaviour is not supported when by_graph_id is True..."
            )

        dataset_length = len(self)
        num_train = int(train_pct * dataset_length)

        if validation_pct > 0:
            num_test = int(test_pct * dataset_length)
            num_validation = dataset_length - num_train - num_test
        else:
            num_test = dataset_length - num_train
            num_validation = 0

        unique_graph_ids = set(
            [
                g.get("id") if hasattr(g, "id") else getattr(g, "graph_id", None)
                for g in self
            ]
        )
        if unique_graph_ids == {None}:
            by_graph_id = False

            warnings.warn(
                f"""No graph_ids available, continuing with by_graph_id=False... If you want to use graph_ids please specify in GraphConverter class""",
                NoGraphIdsWarning,
            )

        if not by_graph_id:
            if random_seed:
                idxs = np.random.RandomState(seed=random_seed).permutation(
                    dataset_length
                )
            else:
                idxs = np.arange(dataset_length)

            if num_validation > 0:
                train_idxs = idxs[:num_train]
                test_idxs = idxs[num_train : num_train + num_test]
                validation_idxs = idxs[
                    num_train + num_test : num_train + num_test + num_validation
                ]

                train_set = self[train_idxs]
                test_set = self[test_idxs]
                validation_set = self[validation_idxs]

                if train_label_ratio is not None:
                    train_set = self._balance_labels(
                        train_set, train_label_ratio, random_seed
                    )
                if test_label_ratio is not None:
                    test_set = self._balance_labels(
                        test_set, test_label_ratio, random_seed
                    )
                if val_label_ratio is not None:
                    validation_set = self._balance_labels(
                        validation_set, val_label_ratio, random_seed
                    )

                return train_set, test_set, validation_set
            else:
                train_idxs = idxs[:num_train]
                test_idxs = idxs[num_train:]

                train_set = self[train_idxs]
                test_set = self[test_idxs]

                if train_label_ratio is not None:
                    train_set = self._balance_labels(
                        train_set, train_label_ratio, random_seed
                    )
                if test_label_ratio is not None:
                    test_set = self._balance_labels(
                        test_set, test_label_ratio, random_seed
                    )

                return train_set, test_set
        else:
            # Get graph IDs in a framework-agnostic way
            graph_ids = np.asarray(
                [
                    (
                        g.get("id")[0]
                        if hasattr(g, "get") and g.get("id") is not None
                        else getattr(g, "graph_id", None)
                    )
                    for g in self
                ]
            )

            if random_seed:
                np.random.seed(random_seed)

            unique_graph_ids_list = sorted(list(unique_graph_ids))
            np.random.shuffle(unique_graph_ids_list)

            test_idxs, train_idxs, validation_idxs = list(), list(), list()

            def __handle_graph_id(i):
                graph_id = unique_graph_ids_list[i]
                unique_graph_ids.remove(graph_id)
                graph_idxs = np.where(graph_ids == graph_id)[0]
                return graph_idxs

            i = 0
            if num_validation > 0:
                while len(validation_idxs) < num_validation:
                    graph_idxs = __handle_graph_id(i)
                    validation_idxs.extend(graph_idxs)
                    i += 1

            while len(test_idxs) < num_test:
                graph_idxs = __handle_graph_id(i)
                test_idxs.extend(graph_idxs)
                i += 1

            train_idxs = np.isin(graph_ids, np.asarray(list(unique_graph_ids)))
            train_idxs = np.where(train_idxs)[0]

            if validation_idxs:
                train_set = self[train_idxs]
                test_set = self[test_idxs]
                validation_set = self[validation_idxs]

                if train_label_ratio is not None:
                    train_set = self._balance_labels(
                        train_set, train_label_ratio, random_seed
                    )
                if test_label_ratio is not None:
                    test_set = self._balance_labels(
                        test_set, test_label_ratio, random_seed
                    )
                if val_label_ratio is not None:
                    validation_set = self._balance_labels(
                        validation_set, val_label_ratio, random_seed
                    )

                return train_set, test_set, validation_set
            else:
                train_set = self[train_idxs]
                test_set = self[test_idxs]

                if train_label_ratio is not None:
                    train_set = self._balance_labels(
                        train_set, train_label_ratio, random_seed
                    )
                if test_label_ratio is not None:
                    test_set = self._balance_labels(
                        test_set, test_label_ratio, random_seed
                    )

                return train_set, test_set

    def _balance_labels(self, dataset, target_ratio, random_seed):
        """Balance a dataset to achieve a target ratio of labels."""
        if random_seed:
            np.random.seed(random_seed)

        if not 0 <= target_ratio <= 1:
            raise ValueError("target_ratio must be between 0 and 1")

        indices_by_label = {0: [], 1: []}

        for i, g in enumerate(dataset):
            # Handle different types of label storage
            if hasattr(g, "y"):
                y_value = g.y
            elif hasattr(g, "get") and g.get("y", None) is not None:
                y_value = g["y"]
            else:
                raise ValueError("Graph has no attribute 'y'...")

            if isinstance(y_value, (np.ndarray, list)):
                if len(y_value) != 1:
                    raise ValueError(
                        f"Expected y to be a single value, but got array of length {len(y_value)}"
                    )
                label = 1 if y_value[0] > 0.5 else 0
            else:
                label = 1 if y_value > 0.5 else 0

            indices_by_label[label].append(i)

        n_zeros = len(indices_by_label[0])
        n_ones = len(indices_by_label[1])
        total = n_zeros + n_ones

        current_ratio = n_ones / total if total > 0 else 0

        if abs(current_ratio - target_ratio) < 0.01:
            return dataset

        if current_ratio > target_ratio:
            target_ones = int(n_zeros * target_ratio / (1 - target_ratio))
            target_zeros = n_zeros
        else:
            target_zeros = int(n_ones * (1 - target_ratio) / target_ratio)
            target_ones = n_ones

        indices_to_keep = []

        if n_zeros > target_zeros:
            sampled_zeros = np.random.choice(
                indices_by_label[0], target_zeros, replace=False
            )
            indices_to_keep.extend(sampled_zeros)
        else:
            indices_to_keep.extend(indices_by_label[0])

        if n_ones > target_ones:
            sampled_ones = np.random.choice(
                indices_by_label[1], target_ones, replace=False
            )
            indices_to_keep.extend(sampled_ones)
        else:
            indices_to_keep.extend(indices_by_label[1])

        np.random.shuffle(indices_to_keep)

        return dataset[indices_to_keep]


# =============================================================================
# SPEKTRAL IMPLEMENTATION
# =============================================================================

try:
    from spektral.data import Dataset, Graph
    from spektral.data.utils import get_spec
    import tensorflow as tf

    _SpektralBase = Dataset
    _HAS_SPEKTRAL = True
except ImportError:
    _SpektralBase = object
    _HAS_SPEKTRAL = False


class SpektralGraphDataset(_GraphDatasetMixin, _SpektralBase, Sequence):
    """
    Spektral-specific GraphDataset implementation.
    """

    def _SpektralGraphDataset__convert(self, data) -> List:
        """Convert incoming data to Spektral Graph format"""
        if not _HAS_SPEKTRAL:
            raise SpektralDependencyError()

        from spektral.data import Graph

        if isinstance(data[0], Graph):
            return [g for i, g in enumerate(data) if i % self.sample == 0]
        elif isinstance(data[0], dict):
            return [
                Graph(
                    x=g["x"],
                    a=g["a"],
                    e=g["e"],
                    y=g["y"],
                    id=g["id"],
                    frame_id=g.get("frame_id", None),
                    object_ids=g.get("object_ids", None),
                    ball_owning_team_id=g.get("ball_owning_team_id", None),
                )
                for i, g in enumerate(data)
                if i % self.sample == 0
            ]
        else:
            raise ValueError(
                f"Cannot convert type {type(data[0])} to Spektral Graph. "
                "Expected Spektral Graph or dict."
            )

    _GraphDatasetMixin__convert = _SpektralGraphDataset__convert

    def read(self) -> List:
        """Return a list of Spektral Graph objects"""
        if not _HAS_SPEKTRAL:
            raise SpektralDependencyError()

        graphs = self._SpektralGraphDataset__convert(self.graphs)
        logging.info(f"Loading {len(graphs)} graphs into SpektralGraphDataset...")
        return graphs

    def dimensions(self) -> Tuple[int, int, int, int, int]:
        """N, F, S, n_out, n"""
        N = max(g.n_nodes for g in self)
        F = self.n_node_features
        S = self.n_edge_features
        n_out = self.n_labels
        n = len(self)
        return (N, F, S, n_out, n)

    @property
    def signature(self):
        """Compute TensorFlow signature for the dataset"""
        if not _HAS_SPEKTRAL:
            raise SpektralDependencyError()

        from spektral.data.utils import get_spec
        import tensorflow as tf

        if len(self.graphs) == 0:
            return None
        signature = {}
        graph = self.graphs[0]

        if graph.x is not None:
            signature["x"] = dict()
            signature["x"]["spec"] = get_spec(graph.x)
            signature["x"]["shape"] = (None, self.n_node_features)
            signature["x"]["dtype"] = tf.as_dtype(graph.x.dtype)

        if graph.a is not None:
            signature["a"] = dict()
            signature["a"]["spec"] = get_spec(graph.a)
            signature["a"]["shape"] = (None, None)
            signature["a"]["dtype"] = tf.as_dtype(graph.a.dtype)

        if graph.e is not None:
            signature["e"] = dict()
            signature["e"]["spec"] = get_spec(graph.e)
            signature["e"]["shape"] = (None, self.n_edge_features)
            signature["e"]["dtype"] = tf.as_dtype(graph.e.dtype)

        if graph.y is not None:
            signature["y"] = dict()
            signature["y"]["spec"] = get_spec(graph.y)
            signature["y"]["shape"] = (self.n_labels,)
            signature["y"]["dtype"] = tf.as_dtype(np.array(graph.y).dtype)

        if hasattr(graph, "g") and graph.g is not None:
            signature["g"] = dict()
            signature["g"]["spec"] = get_spec(graph.g)
            signature["g"]["shape"] = graph.g.shape
            signature["g"]["dtype"] = tf.as_dtype(np.array(graph.g).dtype)

        return signature


# =============================================================================
# PYTORCH GEOMETRIC IMPLEMENTATION
# =============================================================================

try:
    import torch
    from torch_geometric.data import Data

    _HAS_TORCH_GEOMETRIC = True
except ImportError:
    _HAS_TORCH_GEOMETRIC = False


class PyGGraphDataset(_GraphDatasetMixin, Sequence):
    """
    PyTorch Geometric GraphDataset implementation.
    """

    def _PyGGraphDataset__convert(self, data) -> List:
        """Convert incoming data to PyG Data format"""
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "PyTorch Geometric is required for PyGGraphDataset. "
                "Install it using: pip install torch torch-geometric"
            )

        from torch_geometric.data import Data

        if isinstance(data[0], Data):
            return [g for i, g in enumerate(data) if i % self.sample == 0]
        elif isinstance(data[0], dict):
            pyg_graphs = []
            for i, d in enumerate(data):
                if i % self.sample != 0:
                    continue

                # Node features
                x = torch.tensor(d["x"], dtype=torch.float)

                # Get adjacency matrix and convert to edge_index
                a = d["a"].toarray() if hasattr(d["a"], "toarray") else d["a"]
                edge_indices = np.nonzero(a)
                edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)

                # Edge features (already aligned with edges)
                edge_attr = torch.tensor(d["e"], dtype=torch.float)

                # Labels
                y = torch.tensor(d["y"], dtype=torch.long)

                # Create Data object
                graph_data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                )

                # Add custom attributes
                graph_data.graph_id = d.get("id", None)
                graph_data.frame_id = d.get("frame_id", None)
                graph_data.ball_owning_team_id = d.get("ball_owning_team_id", None)
                graph_data.object_ids = d.get("object_ids", None)

                pyg_graphs.append(graph_data)

            return pyg_graphs
        else:
            raise ValueError(
                f"Cannot convert type {type(data[0])} to PyG Data. "
                "Expected PyG Data or dict."
            )

    _GraphDatasetMixin__convert = _PyGGraphDataset__convert

    def read(self) -> List:
        """Return a list of PyG Data objects"""
        if not _HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "PyTorch Geometric is required. "
                "Install it using: pip install torch torch-geometric"
            )

        graphs = self._PyGGraphDataset__convert(self.graphs)
        logging.info(f"Loading {len(graphs)} graphs into PyGGraphDataset...")
        return graphs

    def dimensions(self) -> Tuple[int, int, int, int, int]:
        """N, F, S, n_out, n"""
        N = max(data.num_nodes for data in self)
        F = self[0].num_node_features if len(self) > 0 else 0
        S = self[0].num_edge_features if len(self) > 0 else 0
        n_out = self[0].y.shape[0] if len(self) > 0 else 0
        n = len(self)
        return (N, F, S, n_out, n)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            selected_graphs = [self.graphs[i] for i in idx]
            return PyGGraphDataset(graphs=selected_graphs, sample_rate=1.0)
        else:
            return self.graphs[idx]

    def __repr__(self):
        return f"PyGGraphDataset(n_graphs={len(self)})"


def GraphDataset(
    format: Optional[Literal["spektral", "pyg"]] = "spektral", **kwargs
) -> Union[SpektralGraphDataset, PyGGraphDataset]:
    """
    Factory function that automatically detects and creates the appropriate dataset.

    Args:
        format: Optional format specification ('spektral' or 'pyg').
                Only required when passing dict format graphs or pickle files.
                For Spektral Graph or PyG Data objects, format is auto-detected.
        **kwargs: Arguments passed to the dataset constructor

    Returns:
        SpektralGraphDataset or PyGGraphDataset depending on format

    Examples:
        # Auto-detect from Spektral graphs
        dataset = GraphDataset(graphs=spektral_graph_list)

        # Auto-detect from PyG graphs
        dataset = GraphDataset(graphs=pyg_data_list)

        # Explicit format required for dicts
        dataset = GraphDataset(graphs=dict_list, format='pyg')

        # Explicit format required for pickle files
        dataset = GraphDataset(pickle_file='graphs.pickle.gz', format='spektral')
    """

    def _create_dataset(fmt: str):
        """Helper function to create the appropriate dataset"""
        if fmt.lower() == "spektral":
            return SpektralGraphDataset(**kwargs)
        elif fmt.lower() == "pyg":
            return PyGGraphDataset(**kwargs)
        else:
            raise ValueError(f"format must be 'spektral' or 'pyg', got '{fmt}'")

    # Auto-detect from graphs if provided
    if kwargs.get("graphs", None) is not None:
        graphs = kwargs["graphs"]

        if not isinstance(graphs, list) or len(graphs) == 0:
            raise ValueError("graphs must be a non-empty list")

        first_item = graphs[0]

        # Check if it's a dict - require explicit format
        if isinstance(first_item, dict):
            if format is None:
                raise ValueError(
                    "When passing dict format graphs, you must explicitly specify format='spektral' or format='pyg'"
                )
            return _create_dataset(format)

        # Check if it's a Spektral Graph
        if _HAS_SPEKTRAL:
            from spektral.data import Graph

            if isinstance(first_item, Graph):
                return SpektralGraphDataset(**kwargs)

        # Check if it's a PyG Data object
        if _HAS_TORCH_GEOMETRIC:
            from torch_geometric.data import Data

            if isinstance(first_item, Data):
                return PyGGraphDataset(**kwargs)

        # If we can't detect, raise error
        raise ValueError(
            f"Cannot auto-detect format for type {type(first_item)}. "
            "Please specify format='spektral' or format='pyg' explicitly."
        )

    # For pickle files, require explicit format
    elif (
        kwargs.get("pickle_file", None) is not None
        or kwargs.get("pickle_folder", None) is not None
    ):
        if format is None:
            raise ValueError(
                "When loading from pickle files, you must explicitly specify format='spektral' or format='pyg'"
            )
        return _create_dataset(format)

    else:
        raise ValueError(
            "Must provide either 'graphs', 'pickle_file', or 'pickle_folder'"
        )
