from pathlib import Path
from unravel.soccer import GraphConverter
from unravel.utils import dummy_labels, dummy_graph_ids, CustomSpektralDataset
from unravel.classifiers import CrystalGraphClassifier

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, BinaryAccuracy


from kloppy import skillcorner
from kloppy.domain import TrackingDataset
from typing import List, Dict

from spektral.data import DisjointLoader

import pytest

import numpy as np
import pandas as pd


class TestSpektral:

    @pytest.fixture
    def match_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_match_data.json"

    @pytest.fixture
    def structured_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_structured_data.json.gz"

    @pytest.fixture()
    def dataset(self, match_data: str, structured_data: str) -> TrackingDataset:
        return skillcorner.load(
            raw_data=structured_data,
            meta_data=match_data,
            coordinates="tracab",
            include_empty_frames=False,
            limit=100,
        )

    @pytest.fixture()
    def converter(self, dataset: TrackingDataset) -> GraphConverter:
        gc = GraphConverter(
            dataset=dataset,
            labels=dummy_labels(dataset),
            graph_ids=dummy_graph_ids(dataset),
            ball_carrier_treshold=25.0,
            max_player_speed=12.0,
            max_ball_speed=28.0,
            boundary_correction=None,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            non_potential_receiver_node_value=0.1,
            infer_ball_ownership=True,
            infer_goalkeepers=True,
            random_seed=42,
            pad=False,
            verbose=False,
        )
        gc.node_features.add_x().add_y().add_velocity().add_speed().add_goal_distance().add_goal_angle().add_ball_distance().add_ball_angle().add_team().add_potential_reciever()
        gc.edge_features.add_dist_matrix().add_speed_diff_matrix().add_pos_cos_matrix().add_pos_sin_matrix().add_vel_cos_matrix().add_vel_sin_matrix()
        return gc

    @pytest.fixture()
    def converter_preds(self, dataset: TrackingDataset) -> GraphConverter:
        gc = GraphConverter(
            dataset=dataset,
            prediction=True,
            ball_carrier_treshold=25.0,
            max_player_speed=12.0,
            max_ball_speed=28.0,
            boundary_correction=None,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            non_potential_receiver_node_value=0.1,
            infer_ball_ownership=True,
            infer_goalkeepers=True,
            random_seed=42,
            pad=False,
            verbose=False,
        )
        gc.node_features.add_x().add_y().add_velocity().add_speed().add_goal_distance().add_goal_angle().add_ball_distance().add_ball_angle().add_team().add_potential_reciever()
        gc.edge_features.add_dist_matrix().add_speed_diff_matrix().add_pos_cos_matrix().add_pos_sin_matrix().add_vel_cos_matrix().add_vel_sin_matrix()
        return gc

    def test_training(self, converter: GraphConverter):
        train = CustomSpektralDataset(graphs=converter.to_spektral_graphs())

        cd = converter.to_custom_dataset()
        assert isinstance(cd, CustomSpektralDataset)

        converter.to_pickle("tests/files/test.pickle.gz")

        with pytest.raises(
            ValueError,
            match="Only compressed pickle files of type 'some_file_name.pickle.gz' are supported...",
        ):
            converter.to_pickle("tests/files/test.pickle")

        model = CrystalGraphClassifier()

        assert model.channels == 128
        assert model.drop_out == 0.5
        assert model.n_layers == 3
        assert model.n_out == 1

        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(),
            metrics=[AUC(), BinaryAccuracy()],
        )

        loader_tr = DisjointLoader(train, batch_size=32)
        model.fit(
            loader_tr.load(),
            epochs=1,
            steps_per_epoch=loader_tr.steps_per_epoch,
            verbose=0,
        )

        model_path = "tests/files/models/my-test-gnn"
        model.save(model_path)
        loaded_model = load_model(model_path)

        loader_te = DisjointLoader(train, batch_size=32, epochs=1, shuffle=False)
        pred = model.predict(loader_te.load())

        loader_te = DisjointLoader(train, batch_size=32, epochs=1, shuffle=False)
        loaded_pred = loaded_model.predict(loader_te.load(), use_multiprocessing=True)

        assert np.allclose(pred, loaded_pred, atol=1e-8)

    def test_prediction(self, converter_preds: GraphConverter):
        pred_dataset = CustomSpektralDataset(
            graphs=converter_preds.to_spektral_graphs()
        )
        loader_pred = DisjointLoader(
            pred_dataset, batch_size=32, epochs=1, shuffle=False
        )

        model_path = "tests/files/models/my-test-gnn"
        loaded_model = load_model(model_path)

        preds = loaded_model.predict(loader_pred.load())

        assert not np.any(np.isnan(preds.flatten()))

        df = pd.DataFrame(
            {"frame_id": [x.id for x in pred_dataset], "y": preds.flatten()}
        )

        assert df["frame_id"].iloc[0] == 1524
        assert df["frame_id"].iloc[-1] == 1621
