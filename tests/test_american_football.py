from pathlib import Path

from typing import List

import pytest

import polars as pl
import pandas as pd
import numpy as np

from os.path import join

from datetime import datetime

from spektral.data import Graph

from unravel.american_football import (
    AmericanFootballGraphSettings,
    BigDataBowlDataset,
    AmericanFootballGraphConverter,
    AmericanFootballPitchDimensions,
)
from unravel.american_football.dataset import Constant
from unravel.utils import (
    flatten_to_reshaped_array,
    make_sparse,
    CustomSpektralDataset,
)

from kloppy.domain import Unit


class TestAmericanFootballDataset:

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
    def default_dataset(self, coordinates: str, players: str, plays: str):
        bdb_dataset = BigDataBowlDataset(
            tracking_file_path=coordinates,
            players_file_path=players,
            plays_file_path=plays,
            max_player_speed=8.0,
            max_ball_speed=28.0,
            max_player_acceleration=10.0,
            max_ball_acceleration=10.0,
        )
        bdb_dataset.add_graph_ids(by=["game_id", "play_id"])
        bdb_dataset.add_dummy_labels(by=["game_id", "play_id", "frame_id"])
        return bdb_dataset

    @pytest.fixture
    def non_default_dataset(self, coordinates: str, players: str, plays: str):
        bdb_dataset = BigDataBowlDataset(
            tracking_file_path=coordinates,
            players_file_path=players,
            plays_file_path=plays,
            max_player_speed=12.0,
            max_ball_speed=24.0,
            max_player_acceleration=11.0,
            max_ball_acceleration=12.0,
        )
        bdb_dataset.add_graph_ids(by=["game_id", "play_id"])
        bdb_dataset.add_dummy_labels(by=["game_id", "play_id", "frame_id"])
        return bdb_dataset

    @pytest.fixture
    def raw_dataset(self, coordinates: str):
        return pd.read_csv(coordinates, parse_dates=["time"])

    @pytest.fixture
    def edge_feature_values(self):
        item_idx = 260

        assert_values = {
            "dist": 0.031333127237586675,
            "speed_diff": 0.0725,
            "acc_diff": 0.017000000000000005,
            "pos_cos": 0.21318726919535064,
            "pos_sin": 0.0904411428764118,
            "dir_cos": 0.9999911965824017,
            "dir_sin": 0.5029670423148592,
            "o_cos": 0.9724458937341698,
            "o_sin": 0.6636914093461278,
        }
        return item_idx, assert_values

    @pytest.fixture
    def adj_matrix_values(self):
        return np.asarray(
            [
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

    @pytest.fixture
    def node_feature_values(self):
        item_idx = 6

        assert_values = {
            "x_normed": 0.6679999999999999,
            "y_normed": 0.6906191369606004,
            "uv_sa[0]": 0.0006550334862428781,
            "uv_sa[1]": 0.003179802408809971,
            "s_normed": 0.0025,
            "uv_aa[0]": 0.0012270197205202376,
            "uv_aa[1]": 0.005956459242025523,
            "a_normed": 0.001,
            "dir_sin_normed": 0.9897173160115632,
            "dir_cos_normed": 0.6008808723120034,
            "o_sin_normed": 0.394422899008786,
            "o_cos_normed": 0.9887263812669529,
            "normed_dist_to_goal": 0.31312769316888,
            "normed_dist_to_ball": 0.05817057703598108,
            "normed_dist_to_end_zone": 0.2486666666666667,
            "is_possession_team": 0.0,
            "is_qb": 0.0,
            "is_ball": 0.0,
            "weight_normed": 0.21428571428571427,
            "height_normed": 0.5333333333333333,
        }
        return item_idx, assert_values

    @pytest.fixture
    def arguments(self):
        return dict(
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            attacking_non_qb_node_value=0.1,
            random_seed=42,
            pad=False,
            verbose=False,
        )

    @pytest.fixture
    def non_default_arguments(self):
        return dict(
            self_loop_ball=False,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="dense_ap",
            label_type="binary",
            defending_team_node_value=0.3,
            attacking_non_qb_node_value=0.2,
            random_seed=42,
            pad=False,
            sample_rate=(1 / 2),
        )

    @pytest.fixture
    def gnnc(self, default_dataset, arguments):
        return AmericanFootballGraphConverter(dataset=default_dataset, **arguments)

    @pytest.fixture
    def gnnc_non_default(self, non_default_dataset, non_default_arguments):
        return AmericanFootballGraphConverter(
            dataset=non_default_dataset, **non_default_arguments
        )

    def test_settings(self, gnnc_non_default, non_default_arguments):
        settings = gnnc_non_default.settings
        assert isinstance(settings, AmericanFootballGraphSettings)

        spektral_graphs = gnnc_non_default.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 130
        assert isinstance(data[0], Graph)

        assert settings.pitch_dimensions.pitch_length == 120.0
        assert settings.pitch_dimensions.pitch_width == 53.3
        assert settings.pitch_dimensions.standardized == False
        assert settings.pitch_dimensions.unit == Unit.YARDS
        assert settings.pitch_dimensions.x_dim.max == 60.0
        assert settings.pitch_dimensions.x_dim.min == -60.0
        assert settings.pitch_dimensions.y_dim.max == 26.65
        assert settings.pitch_dimensions.y_dim.min == -26.65
        assert settings.pitch_dimensions.end_zone == 50.0

        assert Constant.BALL == "football"
        assert Constant.QB == "QB"
        assert settings.max_height == 225.0
        assert settings.min_height == 150.0
        assert settings.max_weight == 200.0
        assert settings.min_weight == 60.0

        assert settings.self_loop_ball == non_default_arguments["self_loop_ball"]
        assert (
            settings.adjacency_matrix_connect_type
            == non_default_arguments["adjacency_matrix_connect_type"]
        )
        assert (
            settings.adjacency_matrix_type
            == non_default_arguments["adjacency_matrix_type"]
        )
        assert settings.label_type == non_default_arguments["label_type"]
        assert (
            settings.defending_team_node_value
            == non_default_arguments["defending_team_node_value"]
        )
        assert (
            settings.attacking_non_qb_node_value
            == non_default_arguments["attacking_non_qb_node_value"]
        )

    def test_raw_data(self, raw_dataset: pd.DataFrame):
        row_10 = raw_dataset.loc[10]

        assert row_10["gameId"] == 2021091300
        assert row_10["playId"] == 4845
        assert row_10["nflId"] == 33131
        assert row_10["frameId"] == 11
        assert row_10["time"] == datetime(2021, 9, 14, 3, 54, 18, 700000)
        assert row_10["jerseyNumber"] == 93
        assert row_10["team"] == "BAL"
        assert row_10["playDirection"] == "left"
        assert row_10["x"] == pytest.approx(40.23, rel=1e-9)
        assert row_10["y"] == pytest.approx(21.73, rel=1e-9)
        assert row_10["s"] == pytest.approx(1.5, rel=1e-9)
        assert row_10["a"] == pytest.approx(2.13, rel=1e-9)
        assert row_10["dis"] == pytest.approx(0.19, rel=1e-9)
        assert row_10["o"] == pytest.approx(100.77, rel=1e-9)
        assert row_10["dir"] == pytest.approx(55.29, rel=1e-9)

    def test_dataset_loader(self, default_dataset: tuple):
        assert isinstance(default_dataset, BigDataBowlDataset)
        assert isinstance(default_dataset.data, pl.DataFrame)
        assert isinstance(
            default_dataset.settings.pitch_dimensions, AmericanFootballPitchDimensions
        )

        settings = default_dataset.settings

        assert settings.pitch_dimensions.pitch_length == 120.0
        assert settings.pitch_dimensions.pitch_width == 53.3
        assert settings.pitch_dimensions.x_dim.max == 60.0
        assert settings.pitch_dimensions.y_dim.max == 26.65
        assert settings.pitch_dimensions.standardized == False
        assert settings.pitch_dimensions.unit == Unit.YARDS

        data = default_dataset.data

        assert len(data) == 6049

        row_10 = data[10].to_dict()

        assert row_10["game_id"][0] == 2021091300
        assert row_10["play_id"][0] == 4845
        assert row_10["id"][0] == 33131
        assert row_10["frame_id"][0] == 11
        assert row_10["time"][0] == datetime(2021, 9, 14, 3, 54, 18, 700000)
        assert row_10["jerseyNumber"][0] == 93
        assert row_10["team_id"][0] == "BAL"
        assert row_10["playDirection"][0] == "left"
        assert row_10["x"][0] == pytest.approx(19.770000000000003, rel=1e-9)
        assert row_10["y"][0] == pytest.approx(4.919999999999998, rel=1e-9)
        assert row_10["v"][0] == pytest.approx(1.5, rel=1e-9)
        assert row_10["a"][0] == pytest.approx(2.13, rel=1e-9)
        assert row_10["dis"][0] == pytest.approx(0.19, rel=1e-9)
        assert row_10["o"][0] == pytest.approx(-1.3828243663551074, rel=1e-9)
        assert row_10["dir"][0] == pytest.approx(-2.176600110162128, rel=1e-9)
        assert row_10["event"][0] == None
        assert row_10["position_name"][0] == "DE"
        assert row_10["ball_owning_team_id"][0] == "LV"
        assert row_10["graph_id"][0] == "2021091300-4845"
        assert "label" in data.columns

    def test_conversion(
        self,
        gnnc: AmericanFootballGraphConverter,
        node_feature_values: tuple,
        edge_feature_values: tuple,
        adj_matrix_values: tuple,
    ):
        item_idx_x, node_feature_assert_values = node_feature_values
        item_idx_e, edge_feature_assert_values = edge_feature_values

        results_df = gnnc._convert()

        assert len(results_df) == 263

        row_4 = results_df[4].to_dict()

        x, x0, x1 = row_4["x"][0], row_4["x_shape_0"][0], row_4["x_shape_1"][0]
        a, a0, a1 = row_4["a"][0], row_4["a_shape_0"][0], row_4["a_shape_1"][0]
        e, e0, e1 = row_4["e"][0], row_4["e_shape_0"][0], row_4["e_shape_1"][0]

        assert e0 == 287
        assert e1 == len(edge_feature_assert_values.keys())
        assert x0 == 23
        assert x1 == len(node_feature_assert_values.keys())
        assert a0 == 23
        assert a1 == 23

        x = flatten_to_reshaped_array(x, x0, x1)
        a = flatten_to_reshaped_array(a, a0, a1)
        e = flatten_to_reshaped_array(e, e0, e1)

        assert x.shape == tuple((x0, x1))
        assert a.shape == tuple((a0, a1))
        assert e.shape == tuple((e0, e1))

        assert np.min(a) == 0
        assert np.max(1) == 1

        for idx, node_feature in enumerate(node_feature_assert_values.keys()):
            assert x[item_idx_x][idx] == pytest.approx(
                node_feature_assert_values.get(node_feature), abs=1e-5
            )

        for idx, edge_feature in enumerate(edge_feature_assert_values.keys()):
            assert e[item_idx_e][idx] == pytest.approx(
                edge_feature_assert_values.get(edge_feature), abs=1e-5
            )

        np.testing.assert_array_equal(a, adj_matrix_values)

    def test_to_graph_frames(
        self, gnnc: AmericanFootballGraphConverter, node_feature_values
    ):
        graph_frames = gnnc.to_graph_frames()

        data = graph_frames
        assert len(data) == 263
        assert isinstance(data[0], dict)
        # note: these shape tests fail if we add more features (ie. metabolicpower)

        item_idx_x, node_feature_assert_values = node_feature_values

        x = data[4]["x"]
        assert x.shape == (23, len(node_feature_assert_values.keys()))

        for idx, node_feature in enumerate(node_feature_assert_values.keys()):
            assert x[item_idx_x][idx] == pytest.approx(
                node_feature_assert_values.get(node_feature), abs=1e-5
            )

    def test_to_spektral_graph(
        self,
        gnnc: AmericanFootballGraphConverter,
        node_feature_values: tuple,
        edge_feature_values: tuple,
        adj_matrix_values: tuple,
    ):
        """
        Test navigating (next/prev) through events
        """
        spektral_graphs = gnnc.to_spektral_graphs()

        item_idx_x, node_feature_assert_values = node_feature_values
        item_idx_e, edge_feature_assert_values = edge_feature_values

        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 263
        assert isinstance(data[0], Graph)
        # note: these shape tests fail if we add more features

        x = data[4].x
        assert x.shape == (23, len(node_feature_assert_values.keys()))

        for idx, node_feature in enumerate(node_feature_assert_values.keys()):
            assert x[item_idx_x][idx] == pytest.approx(
                node_feature_assert_values.get(node_feature), abs=1e-5
            )

        e = data[4].e
        for idx, edge_feature in enumerate(edge_feature_assert_values.keys()):
            assert e[item_idx_e][idx] == pytest.approx(
                edge_feature_assert_values.get(edge_feature), abs=1e-5
            )

        def __are_csr_matrices_equal(mat1, mat2):
            return (
                mat1.shape == mat2.shape
                and np.array_equal(mat1.data, mat2.data)
                and np.array_equal(mat1.indices, mat2.indices)
                and np.array_equal(mat1.indptr, mat2.indptr)
            )

        a = data[4].a
        assert __are_csr_matrices_equal(a, make_sparse(adj_matrix_values))

        dataset = CustomSpektralDataset(graphs=spektral_graphs)
        N, F, S, n_out, n = dataset.dimensions()
        assert N == 23
        assert F == len(node_feature_assert_values.keys())
        assert S == 9
        assert n_out == 1
        assert n == 263

    def test_to_pickle(self, gnnc: AmericanFootballGraphConverter):
        """
        Test navigating (next/prev) through events
        """
        pickle_folder = join("tests", "files", "bdb")

        gnnc.to_pickle(file_path=join(pickle_folder, "test_bdb.pickle.gz"))

        data = CustomSpektralDataset(pickle_folder=pickle_folder)

        assert data.n_graphs == 263
