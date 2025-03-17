import pytest
from pathlib import Path
from kloppy import skillcorner, sportec
from kloppy.domain import Ground, TrackingDataset, Orientation
from unravel.soccer import (
    SoccerGraphConverterPolars,
    KloppyPolarsDataset,
    PressingIntensity,
    Constant,
    Column,
    Group,
)
from spektral.data import Graph

class TestPolarFlex:  
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
            max_player_speed=12.0,
            max_player_acceleration=12.0,
            max_ball_speed=13.5,
            max_ball_acceleration=100,
        )
        dataset.add_dummy_labels(by=["game_id", "frame_id"])
        dataset.add_graph_ids(by=["game_id", "frame_id"])
        return dataset
    
    @pytest.fixture()
    def default_converter(
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
    def default_overriden_converter(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:
        my_feature_specs = {
            'node_features':{
                'x_normed': {}, 
                'y_normed': {}, 
                's_normed': {},
                'v_sin_normed': {},
                'v_cos_normed': {},
                'normed_dist_to_goal': {},
                'normed_dist_to_ball': {},
                'is_possession_team': {},
                'is_gk': {},
                'is_ball': {},
                'goal_sin_normed': {},
                'goal_cos_normed': {},
                'ball_sin_normed': {},
                'ball_cos_normed': {},
                'ball_carrier': {}
            },
            'edge_features':{
                'dist_matrix_normed': {'max_distance': 100.0}, 
                'speed_diff_matrix_normed': {}, 
                'pos_cos_matrix': {}, 
                'pos_sin_matrix': {}, 
                'vel_cos_matrix': {}, 
                'vel_sin_matrix': {}
            }
        }
        
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
            feature_specs=my_feature_specs
        )
    
    @pytest.fixture()
    def valid_feature_converter(
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
            feature_specs={
                'node_features':{
                    'x_normed': {}, 
                    'y_normed': {'max_value': 100.0}, 
                    'v_cos_normed': {},
                    'normed_dist_to_goal': {'max_distance': 50.0}
                },
                'edge_features':{
                    'dist_matrix_normed': {'max_distance': 100.0}, 
                    'speed_diff_matrix_normed': {}
                }
            }
        )    
    
    def test_default_features(self, default_converter: SoccerGraphConverterPolars):
        spektral_graphs = default_converter.to_spektral_graphs()
        
        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 384
        assert isinstance(data[0], Graph)

        x = data[0].x
        n_players = x.shape[0]
        assert x.shape == (n_players, 15)
        print(">>>", x[0, 0])
        assert 0.5475659001711429 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.8997899683121747 == pytest.approx(x[0, 4], abs=1e-5)
        assert 0.2941671698429814 == pytest.approx(x[8, 2], abs=1e-5)
        
        e = data[0].e
        print(e)
        assert e.shape == (129, 6)
        assert 0.0 == pytest.approx(e[0, 0], abs=1e-5)
        assert 0.5 == pytest.approx(e[0, 4], abs=1e-5)
        assert 0.28591171233629764 == pytest.approx(e[8, 2], abs=1e-5)

        a = data[0].a
        assert a.shape == (n_players, n_players)
        assert 1.0 == pytest.approx(a[0, 0], abs=1e-5)
        assert 1.0 == pytest.approx(a[0, 4], abs=1e-5)
        assert 0.0 == pytest.approx(a[8, 2], abs=1e-5)
    
    def test_default_overriden_features(self, default_overriden_converter: SoccerGraphConverterPolars):
        spektral_graphs = default_overriden_converter.to_spektral_graphs()
        
        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 384
        assert isinstance(data[0], Graph)

        x = data[0].x
        n_players = x.shape[0]
        assert x.shape == (n_players, 15)
        print(">>>", x[0, 0])
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

    def test_valid_features(self, valid_feature_converter: SoccerGraphConverterPolars):
        spektral_graphs = valid_feature_converter.to_spektral_graphs()
        data = spektral_graphs
        assert data[0].id == "2417-1529"
        assert len(data) == 384
        assert isinstance(data[0], Graph)

        x = data[0].x
        assert x.shape[1] == 4
        assert 0.5475659001711429 == pytest.approx(x[0, 0], abs=1e-5)
        assert 0.2280424804491045 == pytest.approx(x[0, 1], abs=1e-5)
        assert 0.8997899683121747 == pytest.approx(x[0, 2], abs=1e-5)


    def test_empty_feature_specs(self, kloppy_polars_dataset: KloppyPolarsDataset):
        with pytest.raises(ValueError):
            SoccerGraphConverterPolars(
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
                feature_specs={
                    'node_features':{},
                    'edge_features':{}
                }
            )
    
    def test_incorrect_feature_tag(self, kloppy_polars_dataset: KloppyPolarsDataset):
        with pytest.raises(ValueError):
            SoccerGraphConverterPolars(
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
                feature_specs={
                    'player_features':{}
                }
            )
        
    def test_invalid_features(self, kloppy_polars_dataset: KloppyPolarsDataset):
        with pytest.raises(ValueError):
            SoccerGraphConverterPolars(
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
                feature_specs={
                    'node_features':{
                        'x_velocity': {},
                    }
                }
            )
            
    def test_invalid_params(self, kloppy_polars_dataset: KloppyPolarsDataset):
        with pytest.raises(ValueError):
            SoccerGraphConverterPolars(
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
                feature_specs={
                    'edge_features':{
                        'dist_matrix_normed': {'max_value': 100.0},
                    }
                }
            )
        
    def test_invalid_param_type(self, kloppy_polars_dataset: KloppyPolarsDataset):
        with pytest.raises(TypeError):
            SoccerGraphConverterPolars(
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
                feature_specs={
                    'edge_features':{
                        'dist_matrix_normed': {'max_distance': False},
                    }
                }
            )



