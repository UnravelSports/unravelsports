from pathlib import Path
from unravel.soccer import (
    SoccerGraphConverterPolars,
    KloppyPolarsDataset,
    PressingIntensity,
    Constant,
    Column,
    Group,
    rotate_around_line,
)
from unravel.soccer.graphs.features import (
    compute_node_features,
    compute_edge_features,
    compute_adjacency_matrix,
)
from unravel.utils import (
    CustomSpektralDataset,
    reshape_array,
    distances_between_players_normed,
    speed_difference_normed,
    angle_between_players_normed,
    velocity_difference_normed,
    graph_feature,
    x_normed,
    y_normed,
    speeds_normed,
    velocity_components_2d_normed,
    distance_to_goal_normed,
    distance_to_ball_normed,
    is_possession_team,
    is_gk,
    is_ball,
    angle_to_goal_components_2d_normed,
    angle_to_ball_components_2d_normed,
    is_ball_carrier,
)

from kloppy import skillcorner, sportec
from kloppy.domain import Ground, TrackingDataset, Orientation
from typing import List, Dict

from spektral.data import Graph

import pytest

import numpy as np
import numpy.testing as npt

import polars as pl
import json

from os.path import join

import os
from unittest.mock import patch, MagicMock


class TestKloppyPolarsData:
    @pytest.fixture
    def match_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_match_data.json"

    @pytest.fixture
    def structured_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_structured_data.json.gz"

    @pytest.fixture
    def raw_sportec(self, base_dir: Path) -> str:
        return base_dir / "files" / "sportec_tracking.xml"

    @pytest.fixture
    def meta_sportec(self, base_dir: Path) -> str:
        return base_dir / "files" / "sportec_meta.xml"

    @pytest.fixture
    def single_frame_file(self, base_dir: Path) -> str:
        return base_dir / "files" / "test_frame.json"

    @pytest.fixture
    def single_frame_node_feature_result_file(self, base_dir) -> str:
        return base_dir / "files" / "node_features.npy"

    @pytest.fixture
    def single_frame_edge_feature_result_file(self, base_dir) -> str:
        return base_dir / "files" / "edge_features.npy"

    @pytest.fixture
    def single_frame_adj_matrix_result_file(self, base_dir) -> str:
        return base_dir / "files" / "adjacency_matrix.npy"

    @pytest.fixture()
    def single_frame(self, single_frame_file) -> dict:
        with open(single_frame_file, "r") as file:
            loaded_data = json.load(file)

        return {k: np.asarray(v) for k, v in loaded_data.items()}

    @pytest.fixture()
    def single_frame_node_feature_result(
        self, single_frame_node_feature_result_file
    ) -> np.ndarray:
        return np.load(single_frame_node_feature_result_file)

    @pytest.fixture()
    def single_frame_edge_feature_result(
        self, single_frame_edge_feature_result_file
    ) -> np.ndarray:
        return np.load(single_frame_edge_feature_result_file)

    @pytest.fixture()
    def single_frame_adj_matrix_result(
        self, single_frame_adj_matrix_result_file
    ) -> np.ndarray:
        return np.load(single_frame_adj_matrix_result_file)

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
    def kloppy_dataset_sportec(
        self, raw_sportec: str, meta_sportec: str
    ) -> TrackingDataset:
        return sportec.load_tracking(
            raw_data=raw_sportec,
            meta_data=meta_sportec,
            coordinates="secondspectrum",
            only_alive=False,
            limit=500,
        )

    @pytest.fixture()
    def kloppy_polars_sportec_dataset(
        self, kloppy_dataset_sportec: TrackingDataset
    ) -> KloppyPolarsDataset:
        dataset = KloppyPolarsDataset(kloppy_dataset=kloppy_dataset_sportec)
        return dataset

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
    def spc_padding(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:
        ds = kloppy_polars_dataset
        ds.data = ds.data.with_columns(
            [pl.lit(1.0).alias("fake_global_feature_column")]
        )

        ds.data = (
            ds.data.join(
                (
                    ds.data.filter(pl.col("team_id") == Constant.BALL)
                    .select(["frame_id", "x", "y", "z"])
                    .rename({"x": "ball_x", "y": "ball_y", "z": "ball_z"})
                ),
                on=["frame_id"],
                how="left",
            )
            .with_columns(
                [
                    pl.when(pl.col("team_id") != Constant.BALL)
                    .then(
                        (
                            (pl.col("x") - pl.col("ball_x")) ** 2
                            + (pl.col("y") - pl.col("ball_y")) ** 2
                            + (pl.col("z") - pl.col("ball_z")) ** 2
                        ).sqrt()
                    )
                    .otherwise(999.9)
                    .alias("ball_dist"),
                ]
            )
            .drop(["ball_x", "ball_y", "ball_z"])
        )

        return SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset,
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=False,
            pad=True,
            verbose=False,
            sample_rate=(1 / 2),
            global_feature_cols=["fake_global_feature_column"],
        )

    @pytest.fixture()
    def soccer_polars_converter(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:

        return SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset,
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=False,
            pad=False,
            verbose=False,
        )

    @pytest.fixture()
    def soccer_polars_converter_sportec(
        self, kloppy_polars_sportec_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:
        kloppy_polars_sportec_dataset.add_dummy_labels()
        kloppy_polars_sportec_dataset.add_graph_ids()
        return SoccerGraphConverterPolars(dataset=kloppy_polars_sportec_dataset)

    @pytest.fixture()
    def soccer_polars_converter_graph_and_additional_features(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:

        kloppy_polars_dataset.data = (
            kloppy_polars_dataset.data
            # note, normally you'd join these columns on a frame level
            .with_columns(
                [
                    pl.lit(1).alias("fake_graph_feature_a"),
                    pl.lit(0.12).alias("fake_graph_feature_b"),
                    pl.lit(0.45).alias("fake_additional_feature_a"),
                ]
            )
        )

        @graph_feature(is_custom=True, feature_type="edge")
        def custom_edge_feature(**kwargs):
            return (
                kwargs["fake_additional_feature_a"][None, :]
                + kwargs["fake_additional_feature_a"][:, None]
            )

        @graph_feature(is_custom=True, feature_type="node")
        def custom_node_feature(**kwargs):
            return kwargs["fake_additional_feature_a"]

        return SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset,
            global_feature_cols=["fake_graph_feature_a", "fake_graph_feature_b"],
            additional_feature_cols=["fake_additional_feature_a"],
            edge_feature_funcs=[
                distances_between_players_normed,
                speed_difference_normed,
                angle_between_players_normed,
                velocity_difference_normed,
                custom_edge_feature,
            ],
            node_feature_funcs=[
                x_normed,
                y_normed,
                speeds_normed,
                velocity_components_2d_normed,
                distance_to_goal_normed,
                distance_to_ball_normed,
                is_possession_team,
                is_gk,
                is_ball,
                angle_to_goal_components_2d_normed,
                angle_to_ball_components_2d_normed,
                is_ball_carrier,
                custom_node_feature,
            ],
            chunk_size=2_0000,
            non_potential_receiver_node_value=0.1,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            random_seed=False,
            pad=False,
            verbose=False,
        )

    def test_incorrect_custom_features(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:

        kloppy_polars_dataset.data = (
            kloppy_polars_dataset.data
            # note, normally you'd join these columns on a frame level
            .with_columns(
                [
                    pl.lit(1).alias("fake_graph_feature_a"),
                    pl.lit(0.12).alias("fake_graph_feature_b"),
                    pl.lit(0.45).alias("fake_additional_feature_a"),
                ]
            )
        )

        @graph_feature(is_custom=True, feature_type="node")
        def custom_edge_feature(**kwargs):
            return (
                kwargs["fake_additional_feature_a"][None, :]
                + kwargs["fake_additional_feature_a"][:, None]
            )

        with pytest.raises(Exception):
            SoccerGraphConverterPolars(
                dataset=kloppy_polars_dataset,
                global_feature_cols=["fake_graph_feature_a", "fake_graph_feature_b"],
                additional_feature_cols=["fake_additional_feature_a"],
                edge_feature_funcs=[
                    distances_between_players_normed,
                    speed_difference_normed,
                    angle_between_players_normed,
                    velocity_difference_normed,
                    custom_edge_feature,
                ],
                chunk_size=2_0000,
                non_potential_receiver_node_value=0.1,
                self_loop_ball=True,
                adjacency_matrix_connect_type="ball",
                adjacency_matrix_type="split_by_team",
                label_type="binary",
                defending_team_node_value=0.0,
                random_seed=False,
                pad=False,
                verbose=False,
            )

    def test_incorrect_custom_features_no_decorator(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:

        kloppy_polars_dataset.data = (
            kloppy_polars_dataset.data
            # note, normally you'd join these columns on a frame level
            .with_columns(
                [
                    pl.lit(1).alias("fake_graph_feature_a"),
                    pl.lit(0.12).alias("fake_graph_feature_b"),
                    pl.lit(0.45).alias("fake_additional_feature_a"),
                ]
            )
        )

        def custom_edge_feature(**kwargs):
            return (
                kwargs["fake_additional_feature_a"][None, :]
                + kwargs["fake_additional_feature_a"][:, None]
            )

        with pytest.raises(Exception):
            SoccerGraphConverterPolars(
                dataset=kloppy_polars_dataset,
                global_feature_cols=["fake_graph_feature_a", "fake_graph_feature_b"],
                additional_feature_cols=["fake_additional_feature_a"],
                edge_feature_funcs=[
                    distances_between_players_normed,
                    speed_difference_normed,
                    angle_between_players_normed,
                    velocity_difference_normed,
                    custom_edge_feature,
                ],
                chunk_size=2_0000,
                non_potential_receiver_node_value=0.1,
                self_loop_ball=True,
                adjacency_matrix_connect_type="ball",
                adjacency_matrix_type="split_by_team",
                label_type="binary",
                defending_team_node_value=0.0,
                random_seed=False,
                pad=False,
                verbose=False,
            )

    def test_node_feature_computation(
        self,
        soccer_polars_converter_sportec: SoccerGraphConverterPolars,
        single_frame: dict,
        single_frame_node_feature_result: np.ndarray,
    ):

        d = single_frame
        d["ball_id"] = Constant.BALL
        d["possession_team_id"] = d[Column.BALL_OWNING_TEAM_ID][0]
        d["is_gk"] = np.where(
            d[Column.POSITION_NAME]
            == soccer_polars_converter_sportec.settings.goalkeeper_id,
            True,
            False,
        )
        d["position"] = np.stack((d[Column.X], d[Column.Y], d[Column.Z]), axis=-1)
        d["velocity"] = np.stack((d[Column.VX], d[Column.VY], d[Column.VZ]), axis=-1)

        if len(np.where(d["team_id"] == d["ball_id"])[0]) >= 1:
            ball_index = np.where(d["team_id"] == d["ball_id"])[0]
            ball_position = d["position"][ball_index][0]
        else:
            ball_position = np.asarray([np.nan, np.nan])
            ball_index = 0

        ball_carriers = np.where(d[Column.IS_BALL_CARRIER] == True)[0]
        if len(ball_carriers) == 0:
            ball_carrier_idx = None
        else:
            ball_carrier_idx = ball_carriers[0]

        d["ball_position"] = ball_position
        d["ball_idx"] = ball_index
        d["ball_carrier_idx"] = ball_carrier_idx

        node_features, _ = compute_node_features(
            funcs=soccer_polars_converter_sportec.default_node_feature_funcs,
            opts=soccer_polars_converter_sportec.feature_opts,
            settings=soccer_polars_converter_sportec.settings,
            **d,
        )
        np.testing.assert_allclose(
            node_features, single_frame_node_feature_result, rtol=1e-3
        )

    def test_edge_feature_computation(
        self,
        soccer_polars_converter_sportec: SoccerGraphConverterPolars,
        single_frame: dict,
        single_frame_edge_feature_result: np.ndarray,
        single_frame_adj_matrix_result: np.ndarray,
    ):

        d = single_frame
        d["ball_id"] = Constant.BALL
        d["possession_team_id"] = d[Column.BALL_OWNING_TEAM_ID][0]
        d["is_gk"] = np.where(
            d[Column.POSITION_NAME]
            == soccer_polars_converter_sportec.settings.goalkeeper_id,
            True,
            False,
        )
        d["position"] = np.stack((d[Column.X], d[Column.Y], d[Column.Z]), axis=-1)
        d["velocity"] = np.stack((d[Column.VX], d[Column.VY], d[Column.VZ]), axis=-1)

        if len(np.where(d["team_id"] == d["ball_id"])[0]) >= 1:
            ball_index = np.where(d["team_id"] == d["ball_id"])[0]
            ball_position = d["position"][ball_index][0]
        else:
            ball_position = np.asarray([np.nan, np.nan])
            ball_index = 0

        ball_carriers = np.where(d[Column.IS_BALL_CARRIER] == True)[0]
        if len(ball_carriers) == 0:
            ball_carrier_idx = None
        else:
            ball_carrier_idx = ball_carriers[0]

        d["ball_position"] = ball_position
        d["ball_idx"] = ball_index
        d["ball_carrier_idx"] = ball_carrier_idx

        adjacency_matrix = compute_adjacency_matrix(
            settings=soccer_polars_converter_sportec.settings, **d
        )
        np.testing.assert_allclose(
            adjacency_matrix, single_frame_adj_matrix_result, rtol=1e-3
        )

        edge_features, _ = compute_edge_features(
            adjacency_matrix=adjacency_matrix,
            funcs=soccer_polars_converter_sportec.default_edge_feature_funcs,
            opts=soccer_polars_converter_sportec.feature_opts,
            settings=soccer_polars_converter_sportec.settings,
            **d,
        )

        np.testing.assert_allclose(
            edge_features, single_frame_edge_feature_result, rtol=1e-3
        )

    def test_pi_teams_max_home_away(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="teams", ball_method="max", orient="home_away", speed_threshold=2
        )

        assert isinstance(model.output, pl.DataFrame)
        assert len(model.output) == 21
        assert "game_id" in model.output.columns
        assert "period_id" in model.output.columns
        assert "frame_id" in model.output.columns
        assert "timestamp" in model.output.columns
        assert "time_to_intercept" in model.output.columns
        assert "probability_to_intercept" in model.output.columns
        assert "columns" in model.output.columns
        assert "rows" in model.output.columns

        row = model.output[0]
        assert (
            row["time_to_intercept"].dtype
            == row["probability_to_intercept"].dtype
            == pl.List(pl.List(pl.Float64))
        )
        assert row["rows"].dtype == row["columns"].dtype == pl.List(pl.String)

        assert (
            reshape_array(row["rows"][0]).shape
            == reshape_array(row["columns"][0]).shape
            == (11,)
        )
        assert (
            reshape_array(row["time_to_intercept"][0]).shape
            == reshape_array(row["probability_to_intercept"][0]).shape
            == (11, 11)
        )
        home_team, away_team = kloppy_dataset_sportec.metadata.teams
        assert reshape_array(row["rows"][0])[0] in [
            x.player_id for x in home_team.players
        ]
        assert reshape_array(row["columns"][0])[0] in [
            x.player_id for x in away_team.players
        ]

        assert (
            kloppy_polars_sportec_dataset.data[Column.BALL_OWNING_TEAM_ID][0]
            == home_team.team_id
        )
        assert (
            pytest.approx(reshape_array(row["time_to_intercept"][0])[0][0], abs=1e-5)
            == 2.6428493704618106
        )

    def test_pi_teams_include_home_away(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="teams", ball_method="include", orient="home_away", speed_threshold=2
        )
        row = model.output[0]
        assert reshape_array(row["rows"][0]).shape == (12,)
        assert reshape_array(row["columns"][0]).shape == (11,)
        assert reshape_array(row["time_to_intercept"][0]).shape == (12, 11)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (12, 11)

    def test_pi_teams_exclude_home_away(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="teams", ball_method="exclude", orient="home_away", speed_threshold=2
        )
        row = model.output[0]

        arr = reshape_array(row["probability_to_intercept"][0])
        count = np.count_nonzero(np.isclose(arr, 0.0, atol=1e-5))
        assert count == 121

        assert reshape_array(row["rows"][0]).shape == (11,)
        assert reshape_array(row["columns"][0]).shape == (11,)
        assert reshape_array(row["time_to_intercept"][0]).shape == (11, 11)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (11, 11)

    def test_pi_full_max_home_away(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="full", ball_method="max", orient="home_away", speed_threshold=2
        )
        row = model.output[0]
        assert reshape_array(row["rows"][0]).shape == (22,)
        assert reshape_array(row["columns"][0]).shape == (22,)
        assert reshape_array(row["time_to_intercept"][0]).shape == (22, 22)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (22, 22)

        home_team, away_team = kloppy_dataset_sportec.metadata.teams
        home_player_ids = [x.player_id for x in home_team.players]
        away_player_ids = [x.player_id for x in away_team.players]

        for hp_id in reshape_array(row["rows"][0])[0:11]:
            assert hp_id in home_player_ids

        for ap_id in reshape_array(row["rows"][0])[11:]:
            assert ap_id in away_player_ids

    def test_pi_full_exclude_home_away(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="full", ball_method="exclude", orient="home_away", speed_threshold=2
        )
        row = model.output[0]
        assert reshape_array(row["rows"][0]).shape == (22,)
        assert reshape_array(row["columns"][0]).shape == (22,)
        npt.assert_array_equal(
            reshape_array(row["rows"][0]), reshape_array(row["columns"][0])
        )
        assert reshape_array(row["time_to_intercept"][0]).shape == (22, 22)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (22, 22)

    def test_pi_full_include_home_away(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="full", ball_method="include", orient="home_away", speed_threshold=2
        )
        row = model.output[0]
        assert reshape_array(row["rows"][0]).shape == (23,)
        assert reshape_array(row["columns"][0]).shape == (23,)
        assert reshape_array(row["time_to_intercept"][0]).shape == (23, 23)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (23, 23)

    def test_pi_full_include_ball_owning(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="full",
            ball_method="include",
            orient="ball_owning",
            speed_threshold=2,
        )
        row = model.output[0]
        arr = reshape_array(row["probability_to_intercept"][0])
        count = np.count_nonzero(np.isclose(arr, 0.0, atol=1e-5))
        assert count == 527

        assert reshape_array(row["rows"][0]).shape == (23,)
        assert reshape_array(row["columns"][0]).shape == (23,)
        assert reshape_array(row["time_to_intercept"][0]).shape == (23, 23)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (23, 23)

        home_team, away_team = kloppy_dataset_sportec.metadata.teams
        home_player_ids = [x.player_id for x in home_team.players]
        away_player_ids = [x.player_id for x in away_team.players]

        assert (
            kloppy_polars_sportec_dataset.data[Column.BALL_OWNING_TEAM_ID][0]
            == home_team.team_id
        )

        for hp_id in reshape_array(row["rows"][0])[0:11]:
            assert hp_id in home_player_ids

        for ap_id in reshape_array(row["rows"][0])[11:22]:
            assert ap_id in away_player_ids

        assert reshape_array(row["rows"][0])[22] == Constant.BALL

    def test_pi_full_include_pressing(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="full", ball_method="include", orient="pressing", speed_threshold=2
        )
        row = model.output[0]
        assert reshape_array(row["rows"][0]).shape == (23,)
        assert reshape_array(row["columns"][0]).shape == (23,)
        assert reshape_array(row["time_to_intercept"][0]).shape == (23, 23)
        assert reshape_array(row["probability_to_intercept"][0]).shape == (23, 23)
        home_team, away_team = kloppy_dataset_sportec.metadata.teams
        home_player_ids = [x.player_id for x in home_team.players]
        away_player_ids = [x.player_id for x in away_team.players]

        assert (
            kloppy_polars_sportec_dataset.data[Column.BALL_OWNING_TEAM_ID][0]
            == home_team.team_id
        )

        assert (
            reshape_array(row["rows"][0])[22]
            == reshape_array(row["columns"][0])[22]
            == Constant.BALL
        )

        for ap_id in reshape_array(row["columns"][0])[0:11]:
            assert ap_id in away_player_ids

        for hp_id in reshape_array(row["rows"][0])[11:22]:
            assert hp_id in home_player_ids

    def test_pi_teams_exclude_home_away_speed_0(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="teams", ball_method="exclude", orient="home_away", speed_threshold=0
        )
        row = model.output[0]

        arr = reshape_array(row["probability_to_intercept"][0])
        count = np.count_nonzero(np.isclose(arr, 0.0, atol=1e-5))
        assert count == 33

    def test_pi_full_include_ball_owning_speed_0(
        self,
        kloppy_polars_sportec_dataset: KloppyPolarsDataset,
        kloppy_dataset_sportec: TrackingDataset,
    ):
        assert len(kloppy_dataset_sportec) == 21
        assert len(kloppy_polars_sportec_dataset.data) == 21 * 23

        model = PressingIntensity(dataset=kloppy_polars_sportec_dataset)
        model.fit(
            method="full",
            ball_method="include",
            orient="ball_owning",
            speed_threshold=0,
        )
        row = model.output[0]

        arr = reshape_array(row["probability_to_intercept"][0])
        count = np.count_nonzero(np.isclose(arr, 0.0, atol=1e-5))
        assert count == 117

    def test_padding(self, spc_padding: SoccerGraphConverterPolars):
        spektral_graphs = spc_padding.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 192
        assert isinstance(data[0], Graph)

    def spektral_graph(self, soccer_polars_converter: SoccerGraphConverterPolars):
        """
        Test navigating (next/prev) through events
        """
        spektral_graphs = soccer_polars_converter.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 384
        assert isinstance(data[0], Graph)

        x = data[0].x
        n_players = x.shape[0]
        assert x.shape == (n_players, 15)
        assert 0.5475659001711429 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.8997899683121747 == pytest.approx(x[0, 4], abs=1e-5)
        assert 0.2941671698429814 == pytest.approx(x[8, 2], abs=1e-5)

        e = data[0].e
        assert e.shape == (129, 6)
        assert 0.0 == pytest.approx(e[0, 0], abs=1e-5)
        assert 0.5 == pytest.approx(e[0, 4], abs=1e-5)
        assert 0.28591171233629764 == pytest.approx(e[8, 2], abs=1e-5)

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
        assert n == 384

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=True,
            random_seed=42,
        )
        assert train.n_graphs == 256
        assert test.n_graphs == 64
        assert val.n_graphs == 64

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=False,
            random_seed=42,
        )
        assert train.n_graphs == 256
        assert test.n_graphs == 64
        assert val.n_graphs == 64

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=True,
            random_seed=42,
            test_label_ratio=(1 / 3),
            train_label_ratio=(3 / 4),
            val_label_ratio=(1 / 2),
        )

        assert train.n_graphs == 164
        assert test.n_graphs == 52
        assert val.n_graphs == 62

        train, test = dataset.split_test_train(
            split_train=4, split_test=1, by_graph_id=False, random_seed=42
        )
        assert train.n_graphs == 307
        assert test.n_graphs == 77

        train, test = dataset.split_test_train(
            split_train=4, split_test=5, by_graph_id=False, random_seed=42
        )
        assert train.n_graphs == 170
        assert test.n_graphs == 214

        with pytest.raises(
            NotImplementedError,
            match="Make sure split_train > split_test >= split_validation, other behaviour is not supported when by_graph_id is True...",
        ):
            dataset.split_test_train(
                split_train=4, split_test=5, by_graph_id=True, random_seed=42
            )

    def test_to_spektral_graph_level_features(
        self,
        soccer_polars_converter_graph_and_additional_features: SoccerGraphConverterPolars,
    ):
        """
        Test navigating (next/prev) through events
        """
        soccer_polars_converter_graph_and_additional_features.settings.orientation = (
            Orientation.STATIC_HOME_AWAY
        )

        frame = soccer_polars_converter_graph_and_additional_features.dataset.filter(
            pl.col("graph_id") == "2417-1529"
        )

        ball_index = (
            frame.select(pl.arg_where(pl.col("team_id") == Constant.BALL))
            .to_series()
            .to_list()[0]
        )
        assert len(frame) == 15

        spektral_graphs = (
            soccer_polars_converter_graph_and_additional_features.to_spektral_graphs()
        )

        assert 1 == 1

        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 384
        assert isinstance(data[0], Graph)

        x = data[0].x
        n_players = x.shape[0]

        assert x.shape == (n_players, 18)
        assert 1 - 0.5475659001711429 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.9948105277764999 == pytest.approx(x[0, 4], abs=1e-5)
        assert 0.2941671698429814 == pytest.approx(x[8, 2], abs=1e-5)
        assert 1 == pytest.approx(x[ball_index, 16])  # graph feature
        assert 0.12 == pytest.approx(x[ball_index, 17])  # graph feature
        assert 0 == pytest.approx(x[0, 16])
        assert 0 == pytest.approx(x[13, 17])
        assert 0.45 == pytest.approx(x[0][15])  # custom added node feature

        e = data[0].e
        n_players = e.shape[0]
        assert e.shape == (129, 7)
        assert e[:, 6][0] == 0.90

    def test_line_method(self):
        positions = np.array([[1.0, 1.0], [2.0, 3.0], [0.5, 2.5], [4.0, 1.0]])

        velocities = np.array([[3.0, 2.0], [2.0, 1.0], [1.0, 3.0], [-2.0, 1.5]])

        # Define line (vertical line from (6, 0) to (6, 7))
        line_start = np.array([6.0, 0.0])
        line_end = np.array([6.0, 7.0])

        # Perform the rotation for all vectors
        new_positions, new_velocities, intersections, valid_mask = rotate_around_line(
            positions, velocities, line_start, line_end
        )

        assert new_positions == pytest.approx(
            np.array([[11.0, 7.666666666666668], [10.0, 7.0], [0.5, 2.5], [4.0, 1.0]]),
            rel=1e-5,
            abs=1e-8,
        )

        assert new_velocities == pytest.approx(
            np.array([[-3.0, -2.0], [-2.0, -1.0], [1.0, 3.0], [-2.0, 1.5]]),
            rel=1e-5,
            abs=1e-8,
        )

        assert intersections == pytest.approx(
            np.array([[6.0, 4.333333333333334], [6.0, 5.0], [0.0, 0.0], [0.0, 0.0]]),
            rel=1e-5,
            abs=1e-8,
        )

        assert np.array_equal(valid_mask, np.array([True, True, False, False]))

    def test_plot_graph(self, soccer_polars_converter: SoccerGraphConverterPolars):

        plot_path = join("tests", "files", "plot", "test-1.mp4")
        soccer_polars_converter.plot(
            file_path=plot_path,
            fps=10,
            timestamp=pl.duration(seconds=11, milliseconds=800),
            end_timestamp=pl.duration(seconds=11, milliseconds=1000),
            period_id=1,
            team_color_a="#CD0E61",
            team_color_b="#0066CC",
            ball_color="black",
            color_by="ball_owning",
        )

    def test_plot_png_success(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test successful PNG generation with correct parameters."""
        # Setup test file path
        plot_path = os.path.join("tests", "files", "plot", "test-png.png")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # If the file already exists, remove it to start clean
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # Call the plot function
        soccer_polars_converter.plot(
            file_path=plot_path,
            timestamp=pl.duration(seconds=11, milliseconds=800),
            period_id=1,
            color_by="static_home_away",
        )

        # Check that the file was created
        assert os.path.exists(plot_path)
        assert plot_path.endswith(".png")

    def test_plot_png_no_extension(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test PNG generation when no file extension is provided."""
        # Setup test file path without extension
        plot_path = os.path.join("tests", "files", "plot", "test-no-extension")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # Expected path with .png extension
        expected_path = f"{plot_path}.png"

        # If the file already exists, remove it to start clean
        if os.path.exists(expected_path):
            os.remove(expected_path)

        # Call the plot function
        soccer_polars_converter.plot(
            file_path=plot_path,
            timestamp=pl.duration(seconds=11, milliseconds=800),
            period_id=1,
        )

        # Check that the file was created with .png extension
        assert os.path.exists(expected_path)

    def test_plot_error_only_fps(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test error is raised when only fps is provided without end_timestamp."""
        with pytest.raises(ValueError):
            soccer_polars_converter.plot(
                file_path="output_img.png",
                fps=10,  # Only fps provided
                timestamp=pl.duration(seconds=11, milliseconds=800),
                period_id=1,
            )

    def test_plot_error_empty_selection(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        with pytest.raises(ValueError):
            soccer_polars_converter.plot(
                file_path="output_img.png",
                timestamp=pl.duration(minutes=1, seconds=19),
                period_id=1,
            )

    def test_plot_error_only_end_timestamp(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test error is raised when only end_timestamp is provided without fps."""
        with pytest.raises(ValueError):
            soccer_polars_converter.plot(
                file_path="output_img.png",
                end_timestamp=pl.duration(seconds=11, milliseconds=800),
                timestamp=pl.duration(minutes=1, seconds=18),
                period_id=1,
            )

    def test_plot_error_mp4_extension_without_video_params(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test error when .mp4 extension is used but video parameters are not provided."""
        with pytest.raises(ValueError):
            soccer_polars_converter.plot(
                file_path="output_video.mp4",  # MP4 extension
                timestamp=pl.duration(seconds=11, milliseconds=800),
                period_id=1,
                # Missing both fps and end_timestamp
            )

    def test_plot_error_wrong_extension_for_png(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test error when non-png/mp4 extension is used for image output."""
        with pytest.raises(ValueError):
            soccer_polars_converter.plot(
                file_path="output_img.jpg",  # Using .jpg extension
                timestamp=pl.duration(seconds=11, milliseconds=800),
                period_id=1,
            )

    def test_plot_error_wrong_extension_for_mp4(
        self, soccer_polars_converter: SoccerGraphConverterPolars
    ):
        """Test error when non-mp4 extension is used for video output."""
        with pytest.raises(ValueError):
            soccer_polars_converter.plot(
                file_path="output_video.avi",  # Using .avi extension
                fps=10,
                timestamp=pl.duration(seconds=11, milliseconds=800),
                end_timestamp=pl.duration(seconds=11, milliseconds=900),
                period_id=1,
            )
