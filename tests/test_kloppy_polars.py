from pathlib import Path
from unravel.soccer import (
    SoccerGraphConverterPolars,
    KloppyPolarsDataset,
    PressingIntensity,
    Constant,
    Column,
    Group,
)
from unravel.utils import (
    CustomSpektralDataset,
    reshape_array,
)

from kloppy import skillcorner, sportec
from kloppy.domain import Ground, TrackingDataset, Orientation
from typing import List, Dict

from spektral.data import Graph

import pytest

import numpy as np
import numpy.testing as npt

import polars as pl


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
    def soccer_polars_converter_graph_level_features(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:

        kloppy_polars_dataset.data = (
            kloppy_polars_dataset.data
            # note, normally you'd join these columns on a frame level
            .with_columns(
                [
                    pl.lit(1).alias("fake_graph_feature_a"),
                    pl.lit(0.12).alias("fake_graph_feature_b"),
                ]
            )
        )

        return SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset,
            graph_feature_cols=["fake_graph_feature_a", "fake_graph_feature_b"],
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
        self, soccer_polars_converter_graph_level_features: SoccerGraphConverterPolars
    ):
        """
        Test navigating (next/prev) through events
        """
        frame = soccer_polars_converter_graph_level_features.dataset.filter(
            pl.col("graph_id") == "2417-1529"
        )
        ball_index = (
            frame.select(pl.arg_where(pl.col("team_id") == Constant.BALL))
            .to_series()
            .to_list()[0]
        )
        assert len(frame) == 15

        spektral_graphs = (
            soccer_polars_converter_graph_level_features.to_spektral_graphs()
        )

        assert 1 == 1

        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 384
        assert isinstance(data[0], Graph)

        x = data[0].x
        n_players = x.shape[0]
        assert x.shape == (n_players, 17)
        assert 0.5475659001711429 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.8997899683121747 == pytest.approx(x[0, 4], abs=1e-5)
        assert 0.2941671698429814 == pytest.approx(x[8, 2], abs=1e-5)
        assert 1 == pytest.approx(x[ball_index, 15])
        assert 0.12 == pytest.approx(x[ball_index, 16])
        assert 0 == pytest.approx(x[0, 15])
        assert 0 == pytest.approx(x[13, 16])
