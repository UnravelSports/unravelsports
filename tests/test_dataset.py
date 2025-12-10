import pytest
import numpy as np
from pathlib import Path

# Assuming your module structure
from unravel.utils import GraphDataset
from unravel.utils.objects.graph_dataset import SpektralGraphDataset, PyGGraphDataset


class TestGraphDatasetAutoDetection:
    """Test auto-detection of graph types"""

    @pytest.fixture
    def spektral_graphs(self):
        """Create dummy Spektral graphs"""
        from spektral.data import Graph

        graphs = []
        for i in range(10):
            graphs.append(
                Graph(
                    x=np.random.randn(5, 3),
                    a=np.random.randint(0, 2, (5, 5)),
                    e=np.random.randn(10, 2),
                    y=np.array([i % 2]),
                    id=f"graph_{i}",
                )
            )
        return graphs

    @pytest.fixture
    def pyg_graphs(self):
        """Create dummy PyG Data objects"""
        import torch
        from torch_geometric.data import Data

        graphs = []
        for i in range(10):
            graphs.append(
                Data(
                    x=torch.randn(5, 3),
                    edge_index=torch.randint(0, 5, (2, 10)),
                    edge_attr=torch.randn(10, 2),
                    y=torch.tensor([i % 2]),
                )
            )
            graphs[-1].id = f"graph_{i}"
        return graphs

    @pytest.fixture
    def dict_graphs(self):
        """Create dummy dict format graphs"""
        graphs = []
        for i in range(10):
            graphs.append(
                {
                    "x": np.random.randn(5, 3),
                    "a": np.random.randint(0, 2, (5, 5)),
                    "e": np.random.randn(10, 2),
                    "y": np.array([i % 2]),
                    "id": f"graph_{i}",
                    "frame_id": f"frame_{i}",
                }
            )
        return graphs

    @pytest.mark.spektral
    def test_auto_detect_spektral_graphs(self, spektral_graphs):
        """Test that Spektral graphs are auto-detected"""

        dataset = GraphDataset(graphs=spektral_graphs)

        assert isinstance(dataset, SpektralGraphDataset)
        assert len(dataset) == 10

    def test_auto_detect_pyg_graphs(self, pyg_graphs):
        """Test that PyG Data objects are auto-detected"""

        dataset = GraphDataset(graphs=pyg_graphs)

        assert isinstance(dataset, PyGGraphDataset)
        assert len(dataset) == 10


class TestGraphDatasetExplicitFormat:
    """Test explicit format specification for dicts and pickle files"""

    @pytest.fixture
    def dict_graphs(self):
        """Create dummy dict format graphs"""
        graphs = []
        for i in range(10):
            graphs.append(
                {
                    "x": np.random.randn(5, 3),
                    "a": np.random.randint(0, 2, (5, 5)),
                    "e": np.random.randn(10, 2),
                    "y": np.array([i % 2]),
                    "id": f"graph_{i}",
                    "frame_id": f"frame_{i}",
                }
            )
        return graphs

    @pytest.mark.spektral
    def test_dict_graphs_with_spektral_format(self, dict_graphs):
        """Test creating SpektralGraphDataset from dicts with explicit format"""
        dataset = GraphDataset(graphs=dict_graphs, format="spektral")

        assert isinstance(dataset, SpektralGraphDataset)
        assert len(dataset) == 10
        assert repr(dataset) == "SpektralGraphDataset(n_graphs=10)"

    def test_dict_graphs_with_pyg_format(self, dict_graphs):
        """Test creating PyGGraphDataset from dicts with explicit format"""

        dataset = GraphDataset(graphs=dict_graphs, format="pyg")

        assert isinstance(dataset, PyGGraphDataset)
        assert len(dataset) == 10
        assert repr(dataset) == "PyGGraphDataset(n_graphs=10)"

    @pytest.mark.spektral
    def test_dict_graphs_without_format_raises_error(self, dict_graphs):
        """Test that dicts without format raise an error"""
        assert isinstance(GraphDataset(graphs=dict_graphs), SpektralGraphDataset)

    def test_dict_graphs_without_format_raises_error_pyg(self, dict_graphs):
        """Test that dicts without format raise an error"""
        assert isinstance(
            GraphDataset(graphs=dict_graphs, format="pyg"), PyGGraphDataset
        )

    def test_dict_graphs_with_invalid_format_raises_error(self, dict_graphs):
        """Test that invalid format raises an error"""
        with pytest.raises(ValueError):
            GraphDataset(graphs=dict_graphs, format="invalid")

    @pytest.mark.spektral
    def test_pickle_file_with_spektral_format(self, tmp_path, dict_graphs):
        """Test loading pickle file with spektral format"""
        import gzip
        import pickle

        # Create a pickle file
        pickle_path = tmp_path / "test_graphs.pickle.gz"
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(dict_graphs, f)

        dataset = GraphDataset(pickle_file=str(pickle_path), format="spektral")

        assert isinstance(dataset, SpektralGraphDataset)
        assert len(dataset) == 10

    def test_pickle_file_with_pyg_format(self, tmp_path, dict_graphs):
        """Test loading pickle file with spektral format"""
        import gzip
        import pickle

        # Create a pickle file
        pickle_path = tmp_path / "test_graphs.pickle.gz"
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(dict_graphs, f)

        dataset = GraphDataset(pickle_file=str(pickle_path), format="pyg")

        assert isinstance(dataset, PyGGraphDataset)
        assert len(dataset) == 10

    def test_pickle_file_with_pyg_format(self, tmp_path, dict_graphs):
        """Test loading pickle file with pyg format"""
        import gzip
        import pickle

        # Create a pickle file
        pickle_path = tmp_path / "test_graphs.pickle.gz"
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(dict_graphs, f)

        dataset = GraphDataset(pickle_file=str(pickle_path), format="pyg")

        assert isinstance(dataset, PyGGraphDataset)
        assert len(dataset) == 10

    @pytest.mark.spektral
    def test_pickle_file_without_format_raises_error(self, tmp_path, dict_graphs):
        """Test that pickle file without format raises an error"""
        import gzip
        import pickle

        pickle_path = tmp_path / "test_graphs.pickle.gz"
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(dict_graphs, f)

        assert isinstance(
            GraphDataset(pickle_file=str(pickle_path)), SpektralGraphDataset
        )
        assert isinstance(
            GraphDataset(pickle_file=str(pickle_path), format="pyg"), PyGGraphDataset
        )

    def test_pickle_file_without_format_raises_error_pyg(self, tmp_path, dict_graphs):
        """Test that pickle file without format raises an error"""
        import gzip
        import pickle

        pickle_path = tmp_path / "test_graphs.pickle.gz"
        with gzip.open(pickle_path, "wb") as f:
            pickle.dump(dict_graphs, f)

        assert isinstance(
            GraphDataset(pickle_file=str(pickle_path), format="pyg"), PyGGraphDataset
        )


class TestGraphDatasetSplitting:
    """Test dataset splitting functionality"""

    @pytest.fixture
    def pyg_dataset(self):
        """Create a PyG dataset for testing"""
        import torch
        from torch_geometric.data import Data

        graphs = []
        for i in range(100):
            graphs.append(
                Data(
                    x=torch.randn(5, 3),
                    edge_index=torch.randint(0, 5, (2, 10)),
                    edge_attr=torch.randn(10, 2),
                    y=torch.tensor([i % 2]),
                )
            )
            graphs[-1].id = f"graph_{i}"

        return GraphDataset(graphs=graphs)

    @pytest.fixture
    def spektral_dataset(self):
        """Create a Spektral dataset for testing"""
        from spektral.data import Graph

        graphs = []
        for i in range(100):
            graphs.append(
                Graph(
                    x=np.random.randn(5, 3),
                    a=np.random.randint(0, 2, (5, 5)),
                    e=np.random.randn(10, 2),
                    y=np.array([i % 2]),
                    id=f"graph_{i}",
                )
            )

        return GraphDataset(graphs=graphs)

    def test_pyg_split_test_train(self, pyg_dataset):
        """Test PyG dataset train/test split"""
        train, test = pyg_dataset.split_test_train(0.8, 0.2, random_seed=42)

        assert isinstance(train, PyGGraphDataset)
        assert isinstance(test, PyGGraphDataset)
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(pyg_dataset)

    @pytest.mark.spektral
    def test_spektral_split_test_train(self, spektral_dataset):
        """Test Spektral dataset train/test split"""
        train, test = spektral_dataset.split_test_train(0.8, 0.2, random_seed=42)

        assert isinstance(train, SpektralGraphDataset)
        assert isinstance(test, SpektralGraphDataset)
        assert len(train) == 80
        assert len(test) == 20
        assert len(train) + len(test) == len(spektral_dataset)

    def test_pyg_split_test_train_validation(self, pyg_dataset):
        """Test PyG dataset train/test/validation split"""
        train, test, val = pyg_dataset.split_test_train_validation(
            split_train=0.7, split_test=0.2, split_validation=0.1, random_seed=42
        )

        assert isinstance(train, PyGGraphDataset)
        assert isinstance(test, PyGGraphDataset)
        assert isinstance(val, PyGGraphDataset)
        assert len(train) == 70
        assert len(test) == 20
        assert len(val) == 10
        assert len(train) + len(test) + len(val) == len(pyg_dataset)

    @pytest.mark.spektral
    def test_spektral_split_test_train_validation(self, spektral_dataset):
        """Test Spektral dataset train/test/validation split"""
        train, test, val = spektral_dataset.split_test_train_validation(
            split_train=0.7, split_test=0.2, split_validation=0.1, random_seed=42
        )

        assert isinstance(train, SpektralGraphDataset)
        assert isinstance(test, SpektralGraphDataset)
        assert isinstance(val, SpektralGraphDataset)
        assert len(train) == 70
        assert len(test) == 20
        assert len(val) == 10
        assert len(train) + len(test) + len(val) == len(spektral_dataset)


class TestGraphDatasetEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_graphs_list_raises_error(self):
        """Test that empty graphs list raises an error"""
        with pytest.raises(ValueError):
            GraphDataset(graphs=[])

    def test_graphs_not_list_raises_error(self):
        """Test that non-list graphs raises an error"""
        with pytest.raises(ValueError):
            GraphDataset(graphs="not a list")

    def test_no_input_raises_error(self):
        """Test that no input raises an error"""
        with pytest.raises(ValueError):
            GraphDataset()

    def test_unknown_graph_type_raises_error(self):
        """Test that unknown graph type raises an error"""

        class UnknownGraph:
            pass

        unknown_graphs = [UnknownGraph() for _ in range(10)]

        with pytest.raises(ValueError):
            GraphDataset(graphs=unknown_graphs)


class TestGraphDatasetRepr:
    """Test string representation of datasets"""

    @pytest.mark.spektral
    def test_spektral_repr(self):
        """Test SpektralGraphDataset repr"""
        from spektral.data import Graph

        graphs = [
            Graph(
                x=np.random.randn(5, 3),
                a=np.random.randint(0, 2, (5, 5)),
                e=np.random.randn(10, 2),
                y=np.array([0]),
                id="graph_0",
            )
        ]

        dataset = GraphDataset(graphs=graphs)
        assert repr(dataset) == "SpektralGraphDataset(n_graphs=1)"

    def test_pyg_repr(self):
        """Test PyGGraphDataset repr"""
        import torch
        from torch_geometric.data import Data

        graphs = [
            Data(
                x=torch.randn(5, 3),
                edge_index=torch.randint(0, 5, (2, 10)),
                edge_attr=torch.randn(10, 2),
                y=torch.tensor([0]),
            )
        ]

        dataset = GraphDataset(graphs=graphs)
        assert repr(dataset) == "PyGGraphDataset(n_graphs=1)"
