from pathlib import Path
from unravel.soccer import GraphConverter
from unravel.utils import (
    DefaultTrackingModel,
    dummy_labels,
    dummy_graph_ids,
    CustomSpektralDataset,
    GraphFrame,
)
from unravel.utils.features.node_feature_set import NodeFeatureSet

from kloppy import skillcorner
from kloppy.domain import Ground, TrackingDataset, Orientation
from typing import List, Dict

from spektral.data import Graph

import pytest

import numpy as np


class TestKloppyData:

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
            limit=500,
        )

    @pytest.fixture()
    def gnnc(self, dataset: TrackingDataset) -> GraphConverter:
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
            random_seed=False,
            pad=False,
            verbose=False,
        )
        gc.node_features.add_x().add_y().add_velocity().add_speed().add_goal_distance().add_goal_angle().add_ball_distance().add_ball_angle().add_team().add_potential_reciever()
        gc.edge_features.add_dist_matrix().add_speed_diff_matrix().add_pos_cos_matrix().add_pos_sin_matrix().add_vel_cos_matrix().add_vel_sin_matrix()
        return gc
    
    @pytest.fixture()
    def gnnc_padding(self, dataset: TrackingDataset) -> GraphConverter:
        gc = GraphConverter(
            dataset=dataset,
            labels=dummy_labels(dataset),
            graph_id=1234,
            ball_carrier_treshold=25.0,
            max_player_speed=12.0,
            max_ball_speed=28.0,
            boundary_correction=None,
            self_loop_ball=False,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            non_potential_receiver_node_value=0.1,
            infer_ball_ownership=True,
            infer_goalkeepers=True,
            random_seed=False,
            pad=True,
            verbose=False,
        )
        gc.node_features.add_x().add_y().add_velocity().add_speed().add_goal_distance().add_goal_angle().add_ball_distance().add_ball_angle().add_team().add_potential_reciever()
        gc.edge_features.add_dist_matrix().add_speed_diff_matrix().add_pos_cos_matrix().add_pos_sin_matrix().add_vel_cos_matrix().add_vel_sin_matrix()
        return gc

    @pytest.fixture()
    def gnnc_padding_random(self, dataset: TrackingDataset) -> GraphConverter:
        gc = GraphConverter(
            dataset=dataset,
            labels=dummy_labels(dataset),
            # settings
            ball_carrier_treshold=25.0,
            max_player_speed=12.0,
            max_ball_speed=28.0,
            boundary_correction=None,
            self_loop_ball=False,
            adjacency_matrix_connect_type="ball",
            adjacency_matrix_type="split_by_team",
            label_type="binary",
            defending_team_node_value=0.0,
            non_potential_receiver_node_value=0.1,
            infer_ball_ownership=True,
            infer_goalkeepers=True,
            random_seed=42,
            pad=True,
            verbose=False,
        )
        gc.node_features.add_x().add_y().add_velocity().add_speed().add_goal_distance().add_goal_angle().add_ball_distance().add_ball_angle().add_team().add_potential_reciever()
        gc.edge_features.add_dist_matrix().add_speed_diff_matrix().add_pos_cos_matrix().add_pos_sin_matrix().add_vel_cos_matrix().add_vel_sin_matrix()
        return gc

    def test_conversion(self, gnnc: GraphConverter):
        data, label, frame_id, _ = gnnc._convert(gnnc.dataset[2])

        assert isinstance(data, DefaultTrackingModel)
        assert frame_id == 1525

        assert data.attacking_team == Ground.HOME
        assert data.orientation == Orientation.STATIC_HOME_AWAY
        assert data.attacking_players == data.home_players
        assert data.defending_players == data.away_players

        hp = data.home_players[3]
        assert -19.582426479899993 == pytest.approx(hp.x1, abs=1e-5)
        assert 24.3039460863 == pytest.approx(hp.y1, abs=1e-5)
        assert -19.6022318885 == pytest.approx(hp.x2, abs=1e-5)
        assert 24.1632567814 == pytest.approx(hp.y2, abs=1e-5)
        assert hp.position.shape == (2,)
        np.testing.assert_allclose(
            hp.position, np.asarray([hp.x1, hp.y1]), rtol=1e-4, atol=1e-4
        )
        assert hp.is_gk == False
        assert hp.next_position[0] - hp.position[0]

        assert data.ball_carrier_idx == 1
        assert len(data.home_players) == 6
        assert len(data.away_players) == 4

        defending_team_value_node_idx = 10
        non_potential_receiver_node_idx = 11

        gnn_frame = GraphFrame(
            frame_id=frame_id,
            data=data,
            label=label,
            graph_id="abcdefg",
            settings=gnnc.settings,
            node_features=gnnc.node_features,
            edge_features=gnnc.edge_features,
        )
        x = gnn_frame.graph_data.get("x")
        gid = gnn_frame.graph_data.get("id")

        assert x[9, defending_team_value_node_idx] == 0.0
        assert x[1, non_potential_receiver_node_idx] == 0.1

        assert gid == "abcdefg"
    
    def test_conversion_padding(self, gnnc_padding: GraphConverter):
        data, _, frame_id, graph_id = gnnc_padding._convert(gnnc_padding.dataset[2])

        assert isinstance(data, DefaultTrackingModel)
        assert frame_id == 1525
        assert graph_id == 1234

        assert data.attacking_team == Ground.HOME
        assert data.attacking_players == data.home_players
        assert data.defending_players == data.away_players
        assert data.ball_carrier_idx == 1
        assert len(data.home_players) == 11
        assert len(data.away_players) == 11
    
    def test_to_spektral_graph(self, gnnc: GraphConverter):
        """
        Test navigating (next/prev) through events
        """
        spektral_graphs = gnnc.to_spektral_graphs()

        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 387
        assert isinstance(data[0], Graph)
        # note: these shape tests fail if we add more features (ie. acceleration)

        x = data[0].x
        assert x.shape == (10, 12)
        assert -0.42531483968190475 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.18817876281930843 == pytest.approx(x[0, 5], abs=1e-5)
        assert 0.5614587302341536 == pytest.approx(x[8, 2], abs=1e-5)

        e = data[0].e
        assert e.shape == (60, 7)
        assert 0.0 == pytest.approx(e[0, 0], abs=1e-5)
        assert 0.5 == pytest.approx(e[0, 4], abs=1e-5)
        assert 0.31674592566440973 == pytest.approx(e[8, 2], abs=1e-5)

        a = data[0].a
        assert a.shape == (10, 10)
        assert 1.0 == pytest.approx(a[0, 0], abs=1e-5)
        assert 1.0 == pytest.approx(a[0, 4], abs=1e-5)
        assert 0.0 == pytest.approx(a[8, 2], abs=1e-5)

        dataset = CustomSpektralDataset(graphs=spektral_graphs)
        N, F, S, n_out, n = dataset.dimensions()
        assert N == 21
        assert F == 12
        assert S == 7
        assert n_out == 1
        assert n == 387

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=True,
            random_seed=42,
        )
        assert train.n_graphs == 233
        assert test.n_graphs == 77
        assert val.n_graphs == 77

        train, test, val = dataset.split_test_train_validation(
            split_train=4,
            split_test=1,
            split_validation=1,
            by_graph_id=False,
            random_seed=42,
        )
        assert train.n_graphs == 258
        assert test.n_graphs == 64
        assert val.n_graphs == 65

        train, test = dataset.split_test_train(
            split_train=4, split_test=1, by_graph_id=False, random_seed=42
        )
        assert train.n_graphs == 309
        assert test.n_graphs == 78

        train, test = dataset.split_test_train(
            split_train=4, split_test=5, by_graph_id=False, random_seed=42
        )
        assert train.n_graphs == 172
        assert test.n_graphs == 215

        with pytest.raises(
            NotImplementedError,
            match="Make sure split_train > split_test >= split_validation, other behaviour is not supported when by_graph_id is True...",
        ):
            dataset.split_test_train(
                split_train=4, split_test=5, by_graph_id=True, random_seed=42
            )

    def test_to_spektral_graph_padding_random(
        self, gnnc_padding_random: GraphConverter
    ):
        """
        Test navigating (next/prev) through events
        """
        gnnc_padding_random.to_graph_frames()

        spektral_graphs = [
            g.to_spektral_graph() for g in gnnc_padding_random.graph_frames
        ]

        # with random seed = 42 the permuntation is [15  9  0  8 17 12  1 13  5  2 11 20  3  4 18 16 21 22  7 10 14 19  6]
        assert 1 == 1

        data = spektral_graphs
        assert len(data) == 387
        assert isinstance(data[0], Graph)
        # note: these shape tests fail if we add more features (ie. acceleration)

        x = data[0].x
        assert x.shape == (23, 12)
        assert -0.42531483968190475 == pytest.approx(x[2, 0], abs=1e-5)
        assert 0.18817876281930843 == pytest.approx(x[2, 5], abs=1e-5)
        assert 0.5614587302341536 == pytest.approx(x[20, 2], abs=1e-5)

        e = data[0].e
        assert e.shape == (287, 7)
        assert 0.0 == pytest.approx(e[0, 0], abs=1e-5)
        assert 0.4261188174 == pytest.approx(e[75, 4], abs=1e-5)
        assert 0.31674592566440973 == pytest.approx(e[119, 2], abs=1e-5)

        a = data[0].a
        assert a.shape == (23, 23)
        assert 1.0 == pytest.approx(a[2, 2], abs=1e-5)