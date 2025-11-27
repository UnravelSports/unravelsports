from pathlib import Path
from unravel.soccer import KloppyPolarsDataset, SoccerGraphConverter
from unravel.utils import dummy_labels, dummy_graph_ids, GraphDataset
from unravel.classifiers import PyGLightningCrystalGraphClassifier

import torch
import pytorch_lightning as pyl
from torch_geometric.loader import DataLoader

from kloppy import skillcorner
from kloppy.domain import TrackingDataset
from typing import List, Dict

import pytest

import numpy as np
import pandas as pd

from os.path import join


class TestPyTorchGeometric:
    @pytest.fixture
    def match_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_match_data.json"

    @pytest.fixture
    def structured_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_structured_data.json.gz"

    @pytest.fixture()
    def kloppy_dataset(self, match_data: str, structured_data: str) -> TrackingDataset:
        return skillcorner.load(
            raw_data=structured_data,
            meta_data=match_data,
            coordinates="tracab",
            include_empty_frames=False,
            limit=100,
            only_alive=False,
        )

    @pytest.fixture()
    def kloppy_polars_dataset(
        self, kloppy_dataset: TrackingDataset
    ) -> KloppyPolarsDataset:
        dataset = KloppyPolarsDataset(
            kloppy_dataset=kloppy_dataset,
            ball_carrier_threshold=25.0,
            max_player_speed=12.0,
            max_player_acceleration=12.0,
            max_ball_speed=13.5,
            max_ball_acceleration=100,
        )
        dataset.add_dummy_labels(by=["game_id", "frame_id"], random_seed=42)
        dataset.add_graph_ids(by=["game_id", "frame_id"])
        return dataset

    @pytest.fixture()
    def soccer_converter(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverter:
        return SoccerGraphConverter(
            dataset=kloppy_polars_dataset,
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=42,
            pad=True,
            verbose=False,
        )

    @pytest.fixture()
    def soccer_converter_preds(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverter:
        return SoccerGraphConverter(
            dataset=kloppy_polars_dataset,
            prediction=True,
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=42,
            pad=True,
            verbose=False,
        )

    def test_soccer_training(self, soccer_converter: SoccerGraphConverter):
        # Convert to PyTorch Geometric graphs
        pyg_graphs = soccer_converter.to_pytorch_graphs()
        train = GraphDataset(graphs=pyg_graphs, format="pyg")

        pickle_folder = join("tests", "files", "kloppy")

        soccer_converter.to_pickle(join(pickle_folder, "test.pickle.gz"))

        with pytest.raises(
            ValueError,
            match="Only compressed pickle files of type 'some_file_name.pickle.gz' are supported...",
        ):
            soccer_converter.to_pickle(join(pickle_folder, "test.pickle"))

        # Initialize PyTorch Lightning model
        model = PyGLightningCrystalGraphClassifier(
            n_layers=3, channels=128, drop_out=0.5, n_out=1
        )

        assert model.model.n_layers == 3
        assert model.model.channels == 128
        assert model.model.drop_out == 0.5
        assert model.model.n_out == 1

        # Create DataLoader for training and validation
        loader_tr = DataLoader(train, batch_size=32, shuffle=True)
        loader_val = DataLoader(train, batch_size=32, shuffle=False)

        # Initialize trainer
        trainer = pyl.Trainer(
            max_epochs=1,
            accelerator="auto",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )

        # Train model with validation data
        trainer.fit(model, loader_tr, loader_val)

        # Save model using the same path as in jupyter notebook
        model_path = join("tests", "files", "models", "my-first-graph-classifier.ckpt")
        trainer.save_checkpoint(model_path)

        # Load model
        loaded_model = PyGLightningCrystalGraphClassifier.load_from_checkpoint(
            model_path
        )

        # Create test loader
        loader_te = DataLoader(train, batch_size=32, shuffle=False)

        # Make predictions with original model
        trainer_pred = pyl.Trainer(
            accelerator="auto", logger=False, enable_progress_bar=False
        )
        pred = trainer_pred.predict(model, loader_te)
        pred = torch.cat(pred).cpu().numpy()

        # Make predictions with loaded model
        loader_te = DataLoader(train, batch_size=32, shuffle=False)
        loaded_pred = trainer_pred.predict(loaded_model, loader_te)
        loaded_pred = torch.cat(loaded_pred).cpu().numpy()

        assert np.allclose(pred, loaded_pred, atol=1e-6)

    def test_soccer_prediction(self, soccer_converter_preds: SoccerGraphConverter):
        # Convert to PyTorch Geometric graphs
        pyg_graphs = soccer_converter_preds.to_pytorch_graphs()
        pred_dataset = GraphDataset(graphs=pyg_graphs, format="pyg")

        # Create DataLoader
        loader_pred = DataLoader(pred_dataset, batch_size=32, shuffle=False)

        # Load model using the same path as in jupyter notebook
        model_path = join("tests", "files", "models", "my-first-graph-classifier.ckpt")
        loaded_model = PyGLightningCrystalGraphClassifier.load_from_checkpoint(
            model_path
        )

        # Make predictions
        trainer = pyl.Trainer(
            accelerator="auto", logger=False, enable_progress_bar=False
        )
        preds = trainer.predict(loaded_model, loader_pred)
        preds = torch.cat(preds).cpu().numpy()

        assert not np.any(np.isnan(preds.flatten()))

        # Get graph IDs from the dataset
        graph_ids = []
        for i in range(len(pred_dataset)):
            graph = pred_dataset.graphs[i]
            graph_ids.append(graph.id)

        df = pd.DataFrame({"frame_id": graph_ids, "y": preds.flatten()}).sort_values(
            by=["frame_id"]
        )

        assert df["frame_id"].iloc[0] == "2417-1524"
        assert df["frame_id"].iloc[-1] == "2417-1621"
