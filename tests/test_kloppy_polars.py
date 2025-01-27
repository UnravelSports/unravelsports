from pathlib import Path
from unravel.soccer import SoccerGraphConverterPolars, KloppyPolarsDataset
from unravel.utils import (
    dummy_labels,
    dummy_graph_ids,
    CustomSpektralDataset,
)

from kloppy import skillcorner
from kloppy.domain import Ground, TrackingDataset, Orientation
from typing import List, Dict

from spektral.data import Graph

import pytest

import numpy as np


class TestKloppyPolarsData:
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
            limit=500,
        )

    @pytest.fixture()
    def kloppy_polars_dataset(
        self, kloppy_dataset: TrackingDataset
    ) -> KloppyPolarsDataset:
        dataset = KloppyPolarsDataset(
            kloppy_dataset=kloppy_dataset,
            ball_carrier_threshold=25.0,
        )
        dataset.load()
        dataset.add_dummy_labels(by=["game_id", "frame_id"])
        dataset.add_graph_ids(by=["game_id", "frame_id"])
        return dataset

    @pytest.fixture()
    def spc_padding(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:
        return SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset,
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            max_player_speed=12.0,
            max_player_acceleration=12.0,
            max_ball_speed=13.5,
            max_ball_acceleration=100,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=False,
            pad=True,
            verbose=False,
        )

    @pytest.fixture()
    def soccer_polars_converter(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:

        return SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset,
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            max_player_speed=12.0,
            max_player_acceleration=12.0,
            max_ball_speed=13.5,
            max_ball_acceleration=100,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=False,
            pad=False,
            verbose=False,
        )

    def test_padding(self, spc_padding: SoccerGraphConverterPolars):
        spektral_graphs = spc_padding.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 384
        assert isinstance(data[0], Graph)

    def test_to_spektral_graph(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """
        Test navigating (next/prev) through events
        """
        spektral_graphs = soccer_polars_converter.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 489
        assert isinstance(data[0], Graph)

        x = data[0].x
        n_players = x.shape[0]
        assert x.shape == (n_players, 15)
        assert 0.4524340998288571 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.9948105277764999 == pytest.approx(x[0, 4], abs=1e-5)
        assert 0.2941671698429814 == pytest.approx(x[8, 2], abs=1e-5)

        e = data[0].e
        assert e.shape == (129, 6)
        assert 0.0 == pytest.approx(e[0, 0], abs=1e-5)
        assert 0.5 == pytest.approx(e[0, 4], abs=1e-5)
        assert 0.7140882876637022 == pytest.approx(e[8, 2], abs=1e-5)

        a = data[0].a
        assert a.shape == (n_players, n_players)
        assert 1.0 == pytest.approx(a[0, 0], abs=1e-5)
        assert 1.0 == pytest.approx(a[0, 4], abs=1e-5)
        assert 0.0 == pytest.approx(a[8, 2], abs=1e-5)

        dataset = CustomSpektralDataset(graphs=spektral_graphs)
        N, F, S, n_out, n = dataset.dimensions()
        assert N == 20
        assert F == 15
        assert S == 6
        assert n_out == 1
        assert n == 489

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=True,
            random_seed=42,
        )
        assert train.n_graphs == 326
        assert test.n_graphs == 81
        assert val.n_graphs == 82

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=False,
            random_seed=42,
        )
        assert train.n_graphs == 326
        assert test.n_graphs == 81
        assert val.n_graphs == 82

        train, test = dataset.split_test_train(
            split_train=4, split_test=1, by_graph_id=False, random_seed=42
        )
        assert train.n_graphs == 391
        assert test.n_graphs == 98

        train, test = dataset.split_test_train(
            split_train=4, split_test=5, by_graph_id=False, random_seed=42
        )
        assert train.n_graphs == 217
        assert test.n_graphs == 272

        with pytest.raises(
            NotImplementedError,
            match="Make sure split_train > split_test >= split_validation, other behaviour is not supported when by_graph_id is True...",
        ):
            dataset.split_test_train(
                split_train=4, split_test=5, by_graph_id=True, random_seed=42
            )
