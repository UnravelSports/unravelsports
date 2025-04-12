import pytest
from pathlib import Path
import json
from kloppy import skillcorner, sportec
from kloppy.domain import Ground, TrackingDataset, Orientation
from unravel.soccer import SoccerGraphConverterPolars, KloppyPolarsDataset
from spektral.data import Graph


class TestPolarFlex:
    @pytest.fixture
    def match_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_match_data.json"

    @pytest.fixture
    def structured_data(self, base_dir: Path) -> str:
        return base_dir / "files" / "skillcorner_structured_data.json.gz"

    @pytest.fixture
    def feature_specs_file(self, base_dir: Path) -> str:
        return base_dir / "files" / "default_feature_specs.json"

    @pytest.fixture()
    def new_feature_specs_file(self, base_dir: Path) -> str:
        return base_dir / "files" / "new_feature_specs.json"

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
        """
        SoccerGraphConverter without any feature specs overriden. The default feature specs are used.
        """

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
        """
        SoccerGraphConverter with feature specs overriden. All the default feature specs are used except that max_distance of dist_matrix_normed is set to 100.0.
        """
        my_feature_specs = {
            "node_features": {
                "x_normed": None,
                "y_normed": None,
                "s_normed": None,
                "v_sin_normed": None,
                "v_cos_normed": None,
                "normed_dist_to_goal": None,
                "normed_dist_to_ball": None,
                "is_possession_team": None,
                "is_gk": None,
                "is_ball": None,
                "goal_sin_normed": None,
                "goal_cos_normed": None,
                "ball_sin_normed": None,
                "ball_cos_normed": None,
                "ball_carrier": None,
            },
            "edge_features": {
                "dist_matrix_normed": {"max_distance": 100.0},
                "speed_diff_matrix_normed": None,
                "pos_cos_matrix": None,
                "pos_sin_matrix": None,
                "vel_cos_matrix": None,
                "vel_sin_matrix": None,
            },
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
            feature_specs=my_feature_specs,
        )

    @pytest.fixture()
    def default_loaded_converter(
        self,
        kloppy_polars_dataset: KloppyPolarsDataset,
        feature_specs_file: str,
        default_converter: SoccerGraphConverterPolars,
    ) -> SoccerGraphConverterPolars:
        """
        SoccerGraphConverter with feature specs loaded from a json file. The default_converter is saved to a json file and then loaded to create a new converter.
        """
        default_converter.save(feature_specs_file)
        converter = SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset, from_json=feature_specs_file
        )
        return converter

    @pytest.fixture()
    def valid_feature_converter(
        self, kloppy_polars_dataset: KloppyPolarsDataset
    ) -> SoccerGraphConverterPolars:
        """
        SoccerGraphConverter with a subset of valid feature specs overriden.
        """
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
                "node_features": {
                    "x_normed": None,
                    "y_normed": {"max_value": 100.0},
                    "v_cos_normed": None,
                    "normed_dist_to_goal": {"max_distance": 50.0},
                },
                "edge_features": {
                    "dist_matrix_normed": {"max_distance": 100.0},
                    "speed_diff_matrix_normed": None,
                },
            },
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

    def test_default_overriden_features(
        self, default_overriden_converter: SoccerGraphConverterPolars
    ):
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

    def test_default_loaded_features(
        self, default_loaded_converter: SoccerGraphConverterPolars
    ):
        spektral_graphs = default_loaded_converter.to_spektral_graphs()

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
                feature_specs={"node_features": {}, "edge_features": {}},
            )

    def test_incorrect_feature_tag(self, kloppy_polars_dataset: KloppyPolarsDataset):
        """
        Tests if the converter raises a Value error when the feature_specs contain an incorrect tag. Here player_features is not a valid tag.
        """
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
                feature_specs={"player_features": {}},
            )

    def test_invalid_features(self, kloppy_polars_dataset: KloppyPolarsDataset):
        """
        Tests if the converter raises a Value error when the feature_specs contain an invalid feature. Here x_velocity is not a valid feature.
        """
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
                    "node_features": {
                        "x_velocity": {},
                    }
                },
            )

    def test_invalid_params(self, kloppy_polars_dataset: KloppyPolarsDataset):
        """
        Tests if the converter raises a Value error when the feature_specs contain an invalid parameter. Here max_value is an incorrect parameter for dist_matrix_normed.
        """
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
                    "edge_features": {
                        "dist_matrix_normed": {"max_value": 100.0},
                    }
                },
            )

    def test_invalid_param_type(self, kloppy_polars_dataset: KloppyPolarsDataset):
        """
        Tests if the converter raises a TypeError when the feature_specs contain an invalid parameter type. Here max_distance should be a string instead of a float.
        """
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
                    "edge_features": {
                        "dist_matrix_normed": {"max_distance": False},
                    }
                },
            )

    def test_default_load_feature_specs(
        self,
        default_converter: SoccerGraphConverterPolars,
        default_loaded_converter: SoccerGraphConverterPolars,
        feature_specs_file: str,
        new_feature_specs_file: str,
    ):
        """
        Tests if the default feature specs are saved correctly from a json file.
        """
        default_converter.save(feature_specs_file)
        default_loaded_converter.save(new_feature_specs_file)

        with open(feature_specs_file, "r") as f1, open(
            new_feature_specs_file, "r"
        ) as f2:
            default_overriden_specs = json.load(f1)
            new_specs = json.load(f2)
            assert default_overriden_specs == new_specs

    def test_overriden_load_feature_specs(
        self,
        kloppy_polars_dataset: KloppyPolarsDataset,
        default_overriden_converter: SoccerGraphConverterPolars,
        feature_specs_file: str,
        new_feature_specs_file: str,
    ):
        """
        Tests if the default overriden converter is saved and loaded correctly.
        """
        default_overriden_converter.save(feature_specs_file)
        converter = SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset, from_json=feature_specs_file
        )
        converter.save(new_feature_specs_file)

        with open(feature_specs_file, "r") as f1, open(
            new_feature_specs_file, "r"
        ) as f2:
            default_overriden_specs = json.load(f1)
            new_specs = json.load(f2)
            assert default_overriden_specs == new_specs

    def test_valid_load_feature_specs(
        self,
        kloppy_polars_dataset: KloppyPolarsDataset,
        valid_feature_converter: SoccerGraphConverterPolars,
        feature_specs_file: str,
        new_feature_specs_file: str,
    ):
        """
        Tests if the valid feature converter is saved and loaded correctly.
        """
        valid_feature_converter.save(feature_specs_file)
        converter = SoccerGraphConverterPolars(
            dataset=kloppy_polars_dataset, from_json=feature_specs_file
        )
        converter.save(new_feature_specs_file)

        with open(feature_specs_file, "r") as f1, open(
            new_feature_specs_file, "r"
        ) as f2:
            default_overriden_specs = json.load(f1)
            new_specs = json.load(f2)
            assert default_overriden_specs == new_specs
