from pathlib import Path
import pytest

import polars as pl

from os.path import join

from datetime import datetime

from spektral.data import Graph

from unravel.american_football import (
    AmericanFootballGraphSettings,
    BigDataBowlDataset,
    AmericanFootballGraphConverter,
    AmericanFootballPitchDimensions,
)
from unravel.utils import (
    add_graph_id_column,
    add_dummy_label_column,
    flatten_to_reshaped_array,
    CustomSpektralDataset,
)

from kloppy.domain import Unit


class TestAmericanFootballDataset:

    @pytest.fixture
    def coordinates(self, base_dir: Path) -> str:
        return base_dir / "files" / "bdb_coords.csv"

    @pytest.fixture
    def players(self, base_dir: Path) -> str:
        return base_dir / "files" / "bdb_players.csv"

    @pytest.fixture
    def plays(self, base_dir: Path) -> str:
        return base_dir / "files" / "bdb_plays.csv"

    @pytest.fixture
    def dataset(self, coordinates: str, players: str, plays: str):
        bdb_dataset = BigDataBowlDataset(
            tracking_file_path=coordinates,
            players_file_path=players,
            plays_file_path=plays,
        )
        bdb_dataset.load()
        bdb_dataset.add_graph_ids(by=["gameId", "playId"], column_name="graph_id")
        bdb_dataset.add_dummy_labels(
            by=["gameId", "playId", "frameId"], column_name="label"
        )
        return bdb_dataset

    @pytest.fixture
    def gnnc(self, dataset):
        return AmericanFootballGraphConverter(
            dataset=dataset.data,
            pitch_dimensions=dataset.pitch_dimensions,
            label_col="label",
            graph_id_col="graph_id",
            ball_carrier_treshold=25.0,
            max_player_speed=8.0,
            max_ball_speed=28.0,
            max_player_acceleration=10.0,
            max_ball_acceleration=10.0,
            self_loop_ball=True,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            non_potential_receiver_node_value=0.1,
            random_seed=False,
            pad=False,
            verbose=False,
        )

    def test_dataset_loader(self, dataset: tuple):
        assert isinstance(dataset.data, pl.DataFrame)
        assert isinstance(dataset.pitch_dimensions, AmericanFootballPitchDimensions)

        data = dataset.data

        assert len(data) == 782

        row_10 = data[10].to_dict()

        assert row_10["gameId"][0] == 2021091300
        assert row_10["playId"][0] == 4845
        assert row_10["nflId"][0] == 33131
        assert row_10["frameId"][0] == 11
        assert row_10["time"][0] == datetime(2021, 9, 14, 3, 54, 18, 700000)
        assert row_10["jerseyNumber"][0] == 93
        assert row_10["team"][0] == "BAL"
        assert row_10["playDirection"][0] == "left"
        assert row_10["x"][0] == pytest.approx(40.23, rel=1e-9)
        assert row_10["y"][0] == pytest.approx(21.73, rel=1e-9)
        assert row_10["s"][0] == pytest.approx(1.5, rel=1e-9)
        assert row_10["a"][0] == pytest.approx(2.13, rel=1e-9)
        assert row_10["dis"][0] == pytest.approx(0.19, rel=1e-9)
        assert row_10["o"][0] == pytest.approx(0.03069629739190649, rel=1e-9)
        assert row_10["dir"][0] == pytest.approx(0.01684229714000729, rel=1e-9)
        assert row_10["event"][0] == "None"
        assert row_10["officialPosition"][0] == "DE"
        assert row_10["possessionTeam"][0] == "LV"
        assert row_10["graph_id"][0] == "2021091300-4845"
        assert "label" in data.columns

        assert dataset.pitch_dimensions.x_dim.max == 120.0
        assert dataset.pitch_dimensions.y_dim.max == 53.3
        assert dataset.pitch_dimensions.standardized == False
        assert dataset.pitch_dimensions.unit == Unit.YARDS

    def test_conversion(self, gnnc: AmericanFootballGraphConverter):
        results_df = gnnc._convert()

        assert len(results_df) == 34

        row_4 = results_df[4].to_dict()

        x, x0, x1 = row_4["x"][0], row_4["x_shape_0"][0], row_4["x_shape_1"][0]
        a, a0, a1 = row_4["a"][0], row_4["a_shape_0"][0], row_4["a_shape_1"][0]
        e, e0, e1 = row_4["e"][0], row_4["e_shape_0"][0], row_4["e_shape_1"][0]

        assert e0 == 287
        assert e1 == 9
        assert x0 == 23
        assert x1 == 16
        assert a0 == 23
        assert a1 == 23

        assert flatten_to_reshaped_array(x, x0, x1)[2][5] == pytest.approx(
            0.0, rel=1e-9
        )
        assert flatten_to_reshaped_array(x, x0, x1)[4][10] == pytest.approx(
            0.5931123282986119, rel=1e-9
        )
        assert flatten_to_reshaped_array(x, x0, x1)[7][14] == pytest.approx(
            0.0, rel=1e-9
        )

        assert flatten_to_reshaped_array(a, a0, a1)[1][1] == pytest.approx(
            1.0, rel=1e-9
        )
        assert flatten_to_reshaped_array(a, a0, a1)[2][14] == pytest.approx(
            0.0, rel=1e-9
        )
        assert flatten_to_reshaped_array(a, a0, a1)[3][17] == pytest.approx(
            0.0, rel=1e-9
        )

        assert flatten_to_reshaped_array(e, e0, e1)[1][0] == pytest.approx(
            2.820017730440715, rel=1e-9
        )
        assert flatten_to_reshaped_array(e, e0, e1)[1][1] == pytest.approx(
            -0.03, rel=1e-9
        )
        assert flatten_to_reshaped_array(e, e0, e1)[2][3] == pytest.approx(
            0.47520783834599784, rel=1e-9
        )
        assert flatten_to_reshaped_array(e, e0, e1)[3][6] == pytest.approx(
            0.4684261032067833, rel=1e-9
        )
        assert flatten_to_reshaped_array(e, e0, e1)[4][8] == pytest.approx(
            0.49649235918273565, rel=1e-9
        )

    def test_to_graph_frames(self, gnnc: AmericanFootballGraphConverter):
        graph_frames = gnnc.to_graph_frames()

        data = graph_frames
        assert len(data) == 34
        assert isinstance(data[0], dict)
        # note: these shape tests fail if we add more features (ie. metabolicpower)

        x = data[0]["x"]
        assert x.shape == (23, 16)
        assert 0.33191666666666664 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.00125 == pytest.approx(x[1, 4], abs=1e-5)
        assert 0.4983690867767475 == pytest.approx(x[11, 8], abs=1e-5)

    def test_to_spektral_graph(self, gnnc: AmericanFootballGraphConverter):
        """
        Test navigating (next/prev) through events
        """
        spektral_graphs = gnnc.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 34
        assert isinstance(data[0], Graph)
        # note: these shape tests fail if we add more features (ie. metabolicpower)

        x = data[0].x
        assert x.shape == (23, 16)
        assert 0.33191666666666664 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.00125 == pytest.approx(x[1, 4], abs=1e-5)
        assert 0.4983690867767475 == pytest.approx(x[11, 8], abs=1e-5)

        e = data[0].e
        assert e.shape == (287, 9)
        assert 0.0 == pytest.approx(e[0, 0], abs=1e-5)
        assert 0.9999723014707707 == pytest.approx(e[1, 4], abs=1e-5)
        assert 0.5 == pytest.approx(e[11, 8], abs=1e-5)

        a = data[0].a
        assert a.shape == (23, 23)
        assert 1.0 == pytest.approx(a[0, 0], abs=1e-5)
        assert 0.0 == pytest.approx(a[1, 4], abs=1e-5)
        assert 0.0 == pytest.approx(a[11, 8], abs=1e-5)

        dataset = CustomSpektralDataset(graphs=spektral_graphs)
        N, F, S, n_out, n = dataset.dimensions()
        assert N == 23
        assert F == 16
        assert S == 9
        assert n_out == 1
        assert n == 34

    def test_to_pickle(self, gnnc: AmericanFootballGraphConverter):
        """
        Test navigating (next/prev) through events
        """
        pickle_folder = join("tests", "files", "bdb")

        gnnc.to_pickle(file_path=join(pickle_folder, "test_bdb.pickle.gz"))

        data = CustomSpektralDataset(pickle_folder=pickle_folder)

        assert data.n_graphs == 34
