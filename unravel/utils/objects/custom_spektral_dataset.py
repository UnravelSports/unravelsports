import logging
import sys
from typing import List, Tuple, Union, Optional

import numpy as np

import random

import gzip
import pickle
from pathlib import Path

import warnings

import tensorflow as tf

from collections.abc import Sequence

from spektral.data import Dataset, Graph
from spektral.data.utils import get_spec

from ..exceptions import NoGraphIdsWarning


# Function to load data from a .pickle.gz file
def load_pickle_gz(file_path):
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


class CustomSpektralDataset(Dataset, Sequence):
    """
    A CustomSpektralDataset is required to use all Spektral funcitonality, see 'spektral.data -> Dataset'
    """

    def __init__(self, **kwargs):
        """
        Constructor to load parameters.
        """
        self._kwargs = kwargs

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

            self.graphs = kwargs["graphs"]
        else:
            raise NotImplementedError(
                "Please provide either 'pickle_folder', 'pickle_file' or 'graphs' as parameter to CustomSpektralDataset"
            )

        super().__init__(**kwargs)

    def __convert(self, data) -> List[Graph]:
        """
        Convert incoming data to correct List[Graph] format
        """
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
                )
                for i, g in enumerate(data)
                if i % self.sample == 0
            ]
        else:
            raise NotImplementedError()

    def read(self) -> List[Graph]:
        """
        Overriding the read function - to return a list of Graph objects
        """
        graphs = self.__convert(self.graphs)

        logging.info(f"Loading {len(graphs)} graphs into CustomSpektralDataset...")

        return graphs

    def add(self, other, verbose: bool = False):
        other = self.__convert(other)

        if verbose:
            logging.info(f"Adding {len(other)} graphs to CustomSpektralDataset...")

        self.graphs = self.graphs + other

    def dimensions(self) -> Tuple[int, int, int, int, int]:
        """
        N = Max number of nodes
        F = Dimensions of Node Features
        S = Dimensions of Edge Features
        n_out = Dimesion of the target
        n = Number of samples in dataset
        """
        N = max(g.n_nodes for g in self)
        F = self.n_node_features
        S = self.n_edge_features
        n_out = self.n_labels
        n = len(self)
        return (N, F, S, n_out, n)

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

        split_train (float): amount of total samples that will go into train set
        split_test (float): amount of total samples that will go into test set.
        split_validation (float): amount of total samples that will go into validation set. Defaults to 0.0.
        by_graph_id (bool): when we want to split the samples by graph_id, such that all graphs with the same id end up in the same train/test/validation set
            set to True. Defaults to False. When set to True the split ratio's will be approximated,
            because we can't be sure to split the graphs exactly according to the ratios.
        random_seed (int, optional): Random seed for reproducibility
        train_label_ratio (float, optional): If provided, balances the training set to have this ratio of labels (0/1).
            Must be between 0 and 1. Defaults to None (keep original distribution).
        test_label_ratio (float, optional): If provided, balances the test set to have this ratio of labels (0/1).
            Must be between 0 and 1. Defaults to None (keep original distribution).
        val_label_ratio (float, optional): If provided, balances the validation set to have this ratio of labels (0/1).
            Must be between 0 and 1. Defaults to None (keep original distribution).

        for an explanation on splitting behaviour when by_graph_id = True
        see: https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn/blob/main/split_sequences.py
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
            [g.get("id") if hasattr(g, "id") else None for g in self]
        )
        if unique_graph_ids == {None}:
            by_graph_id = False

            warnings.warn(
                f"""No graph_ids available, continuing with by_graph_id=False... If you want to use graph_ids please specify in GraphConverter class""",
                NoGraphIdsWarning,
            )

        if not by_graph_id:
            # if we don't use the graph_ids we simply shuffle all indices and return 2 or 3 randomly shuffled datasets
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

                # Apply label balancing if requested
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

                # Apply label balancing if requested
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
            # if we do use the graph_ids we randomly assign all items of a certain graph_id to either
            # val, test or train. We start with validation, because it's assumed to be the smallest dataset.
            graph_ids = np.asarray([g.get("id")[0] for g in self])

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

                # Apply label balancing if requested
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

                # Apply label balancing if requested
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
        """
        Balance a dataset to achieve a target ratio of labels.

        Args:
            dataset: A CustomSpektralDataset containing Graph objects
            target_ratio: Float between 0 and 1, representing the desired ratio of positive labels
                        (e.g., 0.5 for a 50/50 split)

        Returns:
            A balanced subset of the dataset with the desired label ratio
        """
        if random_seed:
            np.random.seed(random_seed)

        if not 0 <= target_ratio <= 1:
            raise ValueError("target_ratio must be between 0 and 1")

        # Identify indices by label
        indices_by_label = {0: [], 1: []}

        for i, g in enumerate(dataset):
            # Handle different types of label storage
            if hasattr(g, "y"):
                if isinstance(g.y, (np.ndarray, list)):
                    # Check that y is not longer than 1 item
                    if len(g.y) != 1:
                        raise ValueError(
                            f"Expected y to be a single value, but got array of length {len(g.y)}"
                        )
                    label = 1 if g.y[0] > 0.5 else 0
                else:
                    label = 1 if g.y > 0.5 else 0
            elif g.get("y", None) is not None:
                # If using dictionary access
                y_value = g["y"]
                if isinstance(y_value, (np.ndarray, list)):
                    # Check that y is not longer than 1 item
                    if len(y_value) != 1:
                        raise ValueError(
                            f"Expected y to be a single value, but got array of length {len(y_value)}"
                        )
                    label = 1 if y_value[0] > 0.5 else 0
                else:
                    label = 1 if y_value > 0.5 else 0
            else:
                raise ValueError("Graph has no attribute 'y'...")

            indices_by_label[label].append(i)

        # Count samples for each class
        n_zeros = len(indices_by_label[0])
        n_ones = len(indices_by_label[1])
        total = n_zeros + n_ones

        # Calculate current ratio
        current_ratio = n_ones / total if total > 0 else 0

        # If already matching target ratio (within 1%), return as is
        if abs(current_ratio - target_ratio) < 0.01:
            return dataset

        # Calculate how many samples we need for each class
        if current_ratio > target_ratio:
            # Too many positives, keep all negatives
            target_ones = int(n_zeros * target_ratio / (1 - target_ratio))
            target_zeros = n_zeros
        else:
            # Too many negatives, keep all positives
            target_zeros = int(n_ones * (1 - target_ratio) / target_ratio)
            target_ones = n_ones

        indices_to_keep = []

        # Keep samples from class 0 (negative)
        if n_zeros > target_zeros:
            sampled_zeros = np.random.choice(
                indices_by_label[0], target_zeros, replace=False
            )
            indices_to_keep.extend(sampled_zeros)
        else:
            indices_to_keep.extend(indices_by_label[0])

        # Keep samples from class 1 (positive)
        if n_ones > target_ones:
            sampled_ones = np.random.choice(
                indices_by_label[1], target_ones, replace=False
            )
            indices_to_keep.extend(sampled_ones)
        else:
            indices_to_keep.extend(indices_by_label[1])

        # Shuffle indices
        np.random.shuffle(indices_to_keep)

        # Return a subset of the dataset using the balanced indices
        return dataset[indices_to_keep]

    @property
    def signature(self):
        """
        This property computes the signature of the dataset, which can be
        passed to `spektral.data.utils.to_tf_signature(signature)` to compute
        the TensorFlow signature.

        The signature includes TensorFlow TypeSpec, shape, and dtype for all
        characteristic matrices of the graphs in the Dataset.
        """
        if len(self.graphs) == 0:
            return None
        signature = {}
        graph = self.graphs[0]  # This is always non-empty

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
