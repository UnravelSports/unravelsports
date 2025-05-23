from pathlib import Path
from unravel.soccer import KloppyPolarsDataset, SoccerGraphConverter
from unravel.american_football import BigDataBowlDataset, AmericanFootballGraphConverter
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

from os.path import join


class TestSpektral:
    @pytest.fixture
    def coordinates(self, base_dir: Path) -> str:
        return base_dir / "files" / "bdb_coords-1.csv"

    @pytest.fixture
    def players(self, base_dir: Path) -> str:
        return base_dir / "files" / "bdb_players-1.csv"

    @pytest.fixture
    def plays(self, base_dir: Path) -> str:
        return base_dir / "files" / "bdb_plays-1.csv"

    @pytest.fixture
    def bdb_dataset(self, coordinates: str, players: str, plays: str):
        bdb_dataset = BigDataBowlDataset(
            tracking_file_path=coordinates,
            players_file_path=players,
            plays_file_path=plays,
            max_player_speed=8.0,
            max_ball_speed=28.0,
            max_player_acceleration=10.0,
            max_ball_acceleration=10.0,
        )
        bdb_dataset.add_graph_ids(by=["gameId", "playId"])
        bdb_dataset.add_dummy_labels(by=["gameId", "playId", "frameId"])
        return bdb_dataset

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
        # return SoccerGraphConverterDeprecated(
        #     dataset=kloppy_dataset,
        #     labels=dummy_labels(kloppy_dataset),
        #     graph_ids=dummy_graph_ids(kloppy_dataset),
        #     ball_carrier_treshold=25.0,
        #     max_player_speed=12.0,
        #     max_ball_speed=28.0,
        #     boundary_correction=None,
        #     self_loop_ball=True,
        #     adjacency_matrix_connect_type="ball",
        #     adjacency_matrix_type="split_by_team",
        #     label_type="binary",
        #     defending_team_node_value=0.0,
        #     non_potential_receiver_node_value=0.1,
        #     infer_ball_ownership=True,
        #     infer_goalkeepers=True,
        #     random_seed=42,
        #     pad=False,
        #     verbose=False,
        # )
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
        # @pytest.fixture()
        # def soccer_converter_preds(
        #     self, kloppy_dataset: TrackingDataset
        # ) -> SoccerGraphConverterDeprecated:
        #     return SoccerGraphConverterDeprecated(
        #         dataset=kloppy_dataset,
        #         prediction=True,
        #         ball_carrier_treshold=25.0,
        #         max_player_speed=12.0,
        #         max_ball_speed=28.0,
        #         boundary_correction=None,
        #         self_loop_ball=True,
        #         adjacency_matrix_connect_type="ball",
        #         adjacency_matrix_type="split_by_team",
        #         label_type="binary",
        #         defending_team_node_value=0.0,
        #         non_potential_receiver_node_value=0.1,
        #         infer_ball_ownership=True,
        #         infer_goalkeepers=True,
        #         random_seed=42,
        #         pad=False,
        #         verbose=False,
        #     )
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

    @pytest.fixture
    def bdb_converter(
        self, bdb_dataset: BigDataBowlDataset
    ) -> AmericanFootballGraphConverter:
        return AmericanFootballGraphConverter(
            dataset=bdb_dataset,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=42,
            pad=False,
            verbose=False,
        )

    @pytest.fixture
    def bdb_converter_preds(
        self, bdb_dataset: BigDataBowlDataset
    ) -> AmericanFootballGraphConverter:
        return AmericanFootballGraphConverter(
            dataset=bdb_dataset,
            prediction=True,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=42,
            pad=False,
            verbose=False,
        )

    def test_soccer_training(self, soccer_converter: SoccerGraphConverter):
        train = CustomSpektralDataset(graphs=soccer_converter.to_spektral_graphs())

        cd = soccer_converter.to_custom_dataset()
        assert isinstance(cd, CustomSpektralDataset)

        pickle_folder = join("tests", "files", "kloppy")

        soccer_converter.to_pickle(join(pickle_folder, "test.pickle.gz"))

        with pytest.raises(
            ValueError,
            match="Only compressed pickle files of type 'some_file_name.pickle.gz' are supported...",
        ):
            soccer_converter.to_pickle(join(pickle_folder, "test.pickle"))

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
        model_path = join("tests", "files", "models", "my-test-gnn")
        model.save(model_path)
        loaded_model = load_model(model_path)

        loader_te = DisjointLoader(train, batch_size=32, epochs=1, shuffle=False)
        pred = model.predict(loader_te.load())

        loader_te = DisjointLoader(train, batch_size=32, epochs=1, shuffle=False)
        loaded_pred = loaded_model.predict(loader_te.load(), use_multiprocessing=True)

        assert np.allclose(pred, loaded_pred, atol=1e-8)

    def test_soccer_prediction(self, soccer_converter_preds: SoccerGraphConverter):
        pred_dataset = CustomSpektralDataset(
            graphs=soccer_converter_preds.to_spektral_graphs()
        )
        loader_pred = DisjointLoader(
            pred_dataset, batch_size=32, epochs=1, shuffle=False
        )
        model_path = join("tests", "files", "models", "my-test-gnn")
        loaded_model = load_model(model_path)

        preds = loaded_model.predict(loader_pred.load())

        assert not np.any(np.isnan(preds.flatten()))

        df = pd.DataFrame(
            {"frame_id": [x.id for x in pred_dataset], "y": preds.flatten()}
        ).sort_values(by=["frame_id"])

        assert df["frame_id"].iloc[0] == "2417-1524"
        assert df["frame_id"].iloc[-1] == "2417-1622"

    def test_bdb_training(self, bdb_converter: AmericanFootballGraphConverter):
        train = CustomSpektralDataset(graphs=bdb_converter.to_spektral_graphs())

        cd = bdb_converter.to_custom_dataset()
        assert isinstance(cd, CustomSpektralDataset)

        pickle_folder = join("tests", "files", "bdb")

        bdb_converter.to_pickle(join(pickle_folder, "test_bdb.pickle.gz"))

        with pytest.raises(
            ValueError,
            match="Only compressed pickle files of type 'some_file_name.pickle.gz' are supported...",
        ):
            bdb_converter.to_pickle(join(pickle_folder, "test_bdb.pickle"))

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

        model_path = join("tests", "files", "models", "my_bdb-test-gnn")
        model.save(model_path)
        loaded_model = load_model(model_path)

        loader_te = DisjointLoader(train, batch_size=32, epochs=1, shuffle=False)
        pred = model.predict(loader_te.load())

        loader_te = DisjointLoader(train, batch_size=32, epochs=1, shuffle=False)
        loaded_pred = loaded_model.predict(loader_te.load(), use_multiprocessing=True)

        assert np.allclose(pred, loaded_pred, atol=1e-8)

    def test_dbd_prediction(self, bdb_converter_preds: AmericanFootballGraphConverter):
        pred_dataset = CustomSpektralDataset(
            graphs=bdb_converter_preds.to_spektral_graphs()
        )
        loader_pred = DisjointLoader(
            pred_dataset, batch_size=32, epochs=1, shuffle=False
        )

        model_path = join("tests", "files", "models", "my_bdb-test-gnn")
        loaded_model = load_model(model_path)

        preds = loaded_model.predict(loader_pred.load())

        assert not np.any(np.isnan(preds.flatten()))

        df = pd.DataFrame(
            {"frame_id": [x.id for x in pred_dataset], "y": preds.flatten()}
        )

        assert df["frame_id"].iloc[0] == "2021091300-4845"
        assert df["frame_id"].iloc[-1] == "2021103108-54"
