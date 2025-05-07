import logging
import sys
import os
import json
from inspect import signature

from dataclasses import dataclass

from typing import List, Union, Dict, Literal, Any, Optional

from kloppy.domain import (
    MetricPitchDimensions,
)
from kloppy.io import FileLike, open_as_file

from spektral.data import Graph

from .graph_settings_pl import GraphSettingsPolars
from ..dataset.kloppy_polars import KloppyPolarsDataset, Column, Group, Constant
from .features import (
    compute_node_features_pl,
    compute_adjacency_matrix_pl,
    compute_edge_features_pl,
    get_node_feature_func_map,
    get_edge_feature_func_map,
    NodeFeatureDefaults,
    EdgeFeatureDefaults,
)
from .exceptions import VersionChecker

from ...utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

DEFAULT_SOCCER_FEATURE_SPECS = {
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
        "dist_matrix_normed": None,
        "speed_diff_matrix_normed": None,
        "pos_cos_matrix": None,
        "pos_sin_matrix": None,
        "vel_cos_matrix": None,
        "vel_sin_matrix": None,
    },
}


@dataclass(repr=True)
class SoccerGraphConverterPolars(DefaultGraphConverter):
    """
    Converts our dataset TrackingDataset into an internal structure

    Attributes:
        dataset (KloppyPolarsDataset): KloppyPolarsDataset created from a Kloppy dataset.
        chunk_size (int): Determines how many Graphs get processed simultanously.
        non_potential_receiver_node_value (float): Value between 0 and 1 to assign to the defing team players
        graph_feature_cols (list[str]): List of columns in the dataset that are Graph level features (e.g. team strength rating, win probabilities etc)
            we want to add to our model. A list of column names corresponding to the Polars dataframe within KloppyPolarsDataset.data
            that are graph level features. They should be joined to the KloppyPolarsDataset.data dataframe such that
            each Group in the group_by has the same value per column. We take the first value of the group, and assign this as a
            "graph level feature" to the ball node.
    """

    dataset: KloppyPolarsDataset = None

    chunk_size: int = 2_0000
    non_potential_receiver_node_value: float = 0.1
    graph_feature_cols: Optional[List[str]] = None
    from_json: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.dataset, KloppyPolarsDataset):
            raise ValueError("dataset should be of type KloppyPolarsDataset...")

        self.sample = 1.0 if self.sample_rate is None else 1.0 / self.sample_rate

        self.pitch_dimensions: MetricPitchDimensions = (
            self.dataset.settings.pitch_dimensions
        )

        if self.from_json is not None:
            configuration = None
            with open_as_file(self.from_json) as f:
                configuration = json.load(f)
            self._load_from_json(configuration)

        self.label_column: str = (
            self.label_col if self.label_col is not None else self.dataset._label_column
        )
        self.graph_id_column: str = (
            self.graph_id_col
            if self.graph_id_col is not None
            else self.dataset._graph_id_column
        )

        self.dataset_checkpoint = self.dataset
        self.dataset = self.dataset.data

        self._sport_specific_checks()
        self.settings = self._apply_graph_settings()

        if self.pad:
            self.dataset = self._apply_padding()
        else:
            self.dataset = self._remove_incomplete_frames()

        self._validate_feature_specs_general()
        self._shuffle()

    def _validate_feature_specs_general(self):
        # Override the feature specs to the default version if they are not provided
        if self.feature_specs == None or self.feature_specs == {}:
            self.feature_specs = DEFAULT_SOCCER_FEATURE_SPECS

        for key in self.feature_specs.keys():
            if key not in ["node_features", "edge_features"]:
                raise ValueError(
                    f"feature_specs should only contain 'node_features' or 'edge_features' as keys. You provided {key}"
                )

        if "node_features" not in self.feature_specs:
            self.feature_specs["node_features"] = {}
        if "edge_features" not in self.feature_specs:
            self.feature_specs["edge_features"] = {}

        if (
            self.feature_specs["node_features"] == {}
            and self.feature_specs["edge_features"] == {}
        ):
            raise ValueError(
                "Please provide feature_specs for either 'node_features' or 'edge_features' or both..."
            )

        self._validate_feature_specs(
            self.feature_specs,
            get_node_feature_func_map,
            NodeFeatureDefaults,
            "node_features",
        )
        self._validate_feature_specs(
            self.feature_specs,
            get_edge_feature_func_map,
            EdgeFeatureDefaults,
            "edge_features",
        )
        self._populate_feature_specs(get_node_feature_func_map, "node_features")
        self._populate_feature_specs(get_edge_feature_func_map, "edge_features")

    def _populate_feature_specs(self, feature_func, feature_tag):
        """
        Populates the feature specs with custom parameters.
        """
        feature_map = feature_func(settings=self.settings)
        for feature, custom_params in self.feature_specs[feature_tag].items():
            params = feature_map[feature]["defaults"].copy()
            if custom_params is not None:
                params.update(custom_params)

            self.feature_specs[feature_tag][feature] = params

    def _load_from_json(self, configuration: dict) -> None:
        """
        Load the configuration from a JSON file.
        Args:
            file_path (str): Path to the JSON file.
        """

        # Read configuration file

        # if configuration is None:
        #     raise ValueError("Configuration file is empty or invalid.")

        # Validate version
        config_version = configuration.get("package_version")
        if not config_version:
            raise ValueError("Configuration file does not specify a version.")

        VersionChecker.check_versioning(config_version)

        # Do not load label_column and graph_id_column from JSON file
        configuration["graph_converter_attributes"].pop("label_column", None)
        configuration["graph_converter_attributes"].pop("graph_id_column", None)

        # validate data cols
        if "dataset_cols" in configuration:
            # check if all columns in the dataset specified in the JSON file are in the dataset
            for col in self.dataset.data.columns:
                if col not in configuration["dataset_cols"]:
                    raise ValueError(f"Column '{col}' is missing in dataset_cols.")

        # validate graph converter attributes
        for key, value in configuration["graph_converter_attributes"].items():
            if key == "dataset":
                print("Dataset is not settable from JSON file.")
            if (
                key == "graph_feature_cols"
                and configuration["graph_converter_attributes"]["graph_feature_cols"]
                is not None
            ):
                # check if graph feature columns exist in the dataset
                for col in configuration["graph_converter_attributes"][
                    "graph_feature_cols"
                ]:
                    if col not in self.dataset.columns:
                        raise ValueError(
                            f"Graph feature column '{col}' not found in dataset columns."
                        )
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid attribute '{key}' in configuration file.")

        if "graph_settings" in configuration:
            graph_settings_dict = configuration["graph_settings"]
            valid_keys = signature(DefaultGraphSettings).parameters.keys()
            filtered_settings = {
                k: v for k, v in graph_settings_dict.items() if k in valid_keys
            }
            self.settings = DefaultGraphSettings(**filtered_settings)

        # validate dataset feature columns
        if "dataset_features" in configuration:
            for key, value in configuration["dataset_features"].items():
                dataset_features = self.dataset.get_features()
                if key in dataset_features:
                    if value != dataset_features[key]:
                        raise ValueError(
                            f"Feature '{key}' in dataset does not match the value in the configuration file."
                        )
                else:
                    raise ValueError(f"Feature '{key}' not found in dataset features.")

            # validate orientation
            if (
                "orientation"
                in configuration["dataset_features"]["settings"]["pitch_dimensions"]
            ):
                if (
                    configuration["dataset_features"]["settings"]["pitch_dimensions"][
                        "orientation"
                    ]
                    != self.dataset.settings.pitch_dimensions.orientation
                ):
                    raise ValueError(
                        f"Orientation in dataset does not match the value in the configuration file."
                    )

    def _validate_feature_specs(
        self, feature_specs: dict, feature_func, feature_defaults, feature_tag
    ):
        """
        Validate feature specs for correct feature names, parameter names and types
        """
        if feature_tag not in feature_specs:
            return
        feature_map = feature_func(settings=self.settings)
        for feature in feature_specs[feature_tag]:
            if feature not in feature_map:
                raise ValueError(
                    f"feature {feature} is not a valid {feature_tag[:4]} feature. Valid features are {list(feature_map.keys())}"
                )
            # if feature_specs[feature_tag][feature] is a boolean, convert it to dictionary
            if isinstance(feature_specs[feature_tag][feature], bool):
                if feature_specs[feature_tag][feature] == False:
                    feature_specs[feature_tag][feature] = {"value": None}
            if isinstance(feature_specs[feature_tag][feature], dict):
                for key, value in feature_specs[feature_tag][feature].items():
                    if key not in feature_map[feature]["defaults"]:
                        raise ValueError(
                            f"{feature_tag[:4]} feature {feature} does not have a key '{key}'. Valid keys are {list(feature_map[feature]['defaults'].keys())}"
                        )

                    expected_type = feature_defaults.__annotations__.get(key)
                    if expected_type and not isinstance(value, expected_type):
                        raise TypeError(
                            f"Feature {feature} key '{key}' should be of type {expected_type}. Instead got {type(value)}"
                        )

    def _shuffle(self):
        if isinstance(self.settings.random_seed, int):
            self.dataset = self.dataset.sample(
                fraction=1.0, seed=self.settings.random_seed
            )
        elif self.settings.random_seed == True:
            self.dataset = self.dataset.sample(fraction=1.0)
        else:
            pass

    def _remove_incomplete_frames(self) -> pl.DataFrame:
        df = self.dataset
        total_frames = len(df.unique(Group.BY_FRAME))

        valid_frames = (
            df.group_by(Group.BY_FRAME)
            .agg(pl.col(Column.TEAM_ID).n_unique().alias("unique_teams"))
            .filter(pl.col("unique_teams") == 3)
            .select(Group.BY_FRAME)
        )
        dropped_frames = total_frames - len(valid_frames.unique(Group.BY_FRAME))
        if dropped_frames > 0 and self.verbose:
            self.__warn_dropped_frames(dropped_frames, total_frames)

        return df.join(valid_frames, on=Group.BY_FRAME)

    def _apply_padding(self) -> pl.DataFrame:
        df = self.dataset

        keep_columns = [
            Column.TIMESTAMP,
            Column.BALL_STATE,
            Column.POSITION_NAME,
            self.label_column,
            self.graph_id_column,
        ]
        empty_columns = [
            Column.OBJECT_ID,
            Column.IS_BALL_CARRIER,
            Column.X,
            Column.Y,
            Column.Z,
            Column.VX,
            Column.VY,
            Column.VZ,
            Column.SPEED,
            Column.AX,
            Column.AY,
            Column.AZ,
            Column.ACCELERATION,
        ]
        group_by_columns = [
            Column.GAME_ID,
            Column.PERIOD_ID,
            Column.FRAME_ID,
            Column.TEAM_ID,
            Column.BALL_OWNING_TEAM_ID,
        ]

        counts = df.group_by(group_by_columns).agg(
            pl.len().alias("count"), *[pl.first(col).alias(col) for col in keep_columns]
        )

        counts = counts.with_columns(
            [
                pl.when(pl.col(Column.TEAM_ID) == Constant.BALL)
                .then(1)
                .when(pl.col(Column.TEAM_ID) == pl.col(Column.BALL_OWNING_TEAM_ID))
                .then(11)
                .otherwise(11)
                .alias("target_length")
            ]
        )

        groups_to_pad = counts.filter(
            pl.col("count") < pl.col("target_length")
        ).with_columns((pl.col("target_length") - pl.col("count")).alias("repeats"))

        if len(groups_to_pad) == 0:
            return df

        padding_rows = []
        for row in groups_to_pad.iter_rows(named=True):
            base_row = {col: row[col] for col in keep_columns + group_by_columns}
            padding_rows.extend([base_row] * row["repeats"])

        padding_df = pl.DataFrame(padding_rows)

        schema = df.schema
        padding_df = padding_df.with_columns(
            [
                pl.lit(0.0 if schema[col] != pl.String else "None")
                .cast(schema[col])
                .alias(col)
                for col in empty_columns
            ]
        )

        padding_df = padding_df.select(df.columns)

        result = pl.concat([df, padding_df], how="vertical")

        total_frames = result.select(Group.BY_FRAME).unique().height

        frame_completeness = (
            result.group_by(Group.BY_FRAME)
            .agg(
                [
                    (pl.col(Column.TEAM_ID).eq(Constant.BALL).sum() == 1).alias(
                        "has_ball"
                    ),
                    (
                        pl.col(Column.TEAM_ID)
                        .eq(pl.col(Column.BALL_OWNING_TEAM_ID))
                        .sum()
                        == 11
                    ).alias("has_owning_team"),
                    (
                        (
                            ~pl.col(Column.TEAM_ID).eq(Constant.BALL)
                            & ~pl.col(Column.TEAM_ID).eq(
                                pl.col(Column.BALL_OWNING_TEAM_ID)
                            )
                        ).sum()
                        == 11
                    ).alias("has_other_team"),
                ]
            )
            .filter(
                pl.col("has_ball")
                & pl.col("has_owning_team")
                & pl.col("has_other_team")
            )
        )

        complete_frames = frame_completeness.height

        dropped_frames = total_frames - complete_frames
        if dropped_frames > 0 and self.verbose:
            self.__warn_dropped_frames(dropped_frames, total_frames)

        return result.join(frame_completeness, on=Group.BY_FRAME, how="inner")

    @staticmethod
    def __warn_dropped_frames(dropped_frames, total_frames):
        import warnings

        warnings.warn(
            f"""Setting pad=True drops frames that do not have at least 1 object for the attacking team, defending team or ball.
            This operation dropped {dropped_frames} incomplete frames out of {total_frames} total frames ({(dropped_frames/total_frames)*100:.2f}%)
            """
        )

    def _apply_graph_settings(self):
        return GraphSettingsPolars(
            pitch_dimensions=self.pitch_dimensions,
            max_player_speed=self.settings.max_player_speed,
            max_ball_speed=self.settings.max_ball_speed,
            max_player_acceleration=self.settings.max_player_acceleration,
            max_ball_acceleration=self.settings.max_ball_acceleration,
            self_loop_ball=self.self_loop_ball,
            adjacency_matrix_connect_type=self.adjacency_matrix_connect_type,
            adjacency_matrix_type=self.adjacency_matrix_type,
            label_type=self.label_type,
            defending_team_node_value=self.defending_team_node_value,
            non_potential_receiver_node_value=self.non_potential_receiver_node_value,
            random_seed=self.random_seed,
            pad=self.pad,
            verbose=self.verbose,
        )

    def _sport_specific_checks(self):
        if not isinstance(self.label_column, str):
            raise Exception("'label_col' should be of type string (str)")

        if not isinstance(self.graph_id_column, str):
            raise Exception("'graph_id_col' should be of type string (str)")

        if not isinstance(self.chunk_size, int):
            raise Exception("chunk_size should be of type integer (int)")

        if not self.label_column in self.dataset.columns and not self.prediction:
            raise Exception(
                "Please specify a 'label_col' and add that column to your 'dataset' or set 'prediction=True' if you want to use the converted dataset to make predictions on."
            )

        if not self.label_column in self.dataset.columns and self.prediction:
            self.dataset = self.dataset.with_columns(
                pl.lit(None).alias(self.label_column)
            )

        if not self.graph_id_column in self.dataset.columns:
            raise Exception(
                "Please specify a 'graph_id_col' and add that column to your 'dataset' ..."
            )

        if self.non_potential_receiver_node_value and not isinstance(
            self.non_potential_receiver_node_value, float
        ):
            raise Exception(
                "'non_potential_receiver_node_value' should be of type float"
            )

    @property
    def __exprs_variables(self):
        exprs_variables = [
            Column.X,
            Column.Y,
            Column.Z,
            Column.SPEED,
            Column.VX,
            Column.VY,
            Column.VZ,
            Column.ACCELERATION,
            Column.AX,
            Column.AY,
            Column.AZ,
            Column.TEAM_ID,
            Column.POSITION_NAME,
            Column.BALL_OWNING_TEAM_ID,
            Column.IS_BALL_CARRIER,
            self.graph_id_column,
            self.label_column,
        ]
        exprs = (
            exprs_variables
            if self.graph_feature_cols is None
            else exprs_variables + self.graph_feature_cols
        )
        return exprs

    def __compute(self, args: List[pl.Series]) -> dict:
        d = {col: args[i].to_numpy() for i, col in enumerate(self.__exprs_variables)}

        if self.graph_feature_cols is not None:
            failed = [
                col
                for col in self.graph_feature_cols
                if not np.all(d[col] == d[col][0])
            ]
            if failed:
                raise ValueError(
                    f"""graph_feature_cols contains multiple different values for a group in the groupby ({Group.BY_FRAME}) selection for the columns {failed}. Make sure each group has the same values per individual column."""
                )

        graph_features = (
            np.asarray([d[col] for col in self.graph_feature_cols]).T[0]
            if self.graph_feature_cols
            else None
        )

        if not np.all(d[self.graph_id_column] == d[self.graph_id_column][0]):
            raise ValueError(
                "graph_id selection contains multiple different values. Make sure each graph_id is unique by at least game_id and frame_id..."
            )

        if not self.prediction and not np.all(
            d[self.label_column] == d[self.label_column][0]
        ):
            raise ValueError(
                """Label selection contains multiple different values for a single selection (group by) of game_id and frame_id, 
                make sure this is not the case. Each group can only have 1 label."""
            )

        ball_carriers = np.where(d[Column.IS_BALL_CARRIER] == True)[0]
        if len(ball_carriers) == 0:
            ball_carrier_idx = None
        else:
            ball_carrier_idx = ball_carriers[0]

        adjacency_matrix = compute_adjacency_matrix_pl(
            team=d[Column.TEAM_ID],
            ball_owning_team=d[Column.BALL_OWNING_TEAM_ID],
            settings=self.settings,
            ball_carrier_idx=ball_carrier_idx,
        )

        velocity = np.stack((d[Column.VX], d[Column.VY]), axis=-1)
        edge_features = compute_edge_features_pl(
            adjacency_matrix=adjacency_matrix,
            p3d=np.stack((d[Column.X], d[Column.Y], d[Column.Z]), axis=-1),
            p2d=np.stack((d[Column.X], d[Column.Y]), axis=-1),
            s=d[Column.SPEED],
            velocity=velocity,
            team=d[Column.TEAM_ID],
            settings=self.settings,
            feature_dict=self.feature_specs["edge_features"],
        )

        node_features = compute_node_features_pl(
            d[Column.X],
            d[Column.Y],
            s=d[Column.SPEED],
            velocity=velocity,
            team=d[Column.TEAM_ID],
            possession_team=d[Column.BALL_OWNING_TEAM_ID],
            is_gk=(d[Column.POSITION_NAME] == self.settings.goalkeeper_id).astype(int),
            ball_carrier=d[Column.IS_BALL_CARRIER],
            graph_features=graph_features,
            settings=self.settings,
            feature_dict=self.feature_specs["node_features"],
        )

        return {
            "e": edge_features.tolist(),
            "x": node_features.tolist(),
            "a": adjacency_matrix.tolist(),
            self.graph_id_column: d[self.graph_id_column][0],
            self.label_column: d[self.label_column][0],
        }

    @property
    def return_dtypes(self):
        return pl.Struct(
            {
                "e": pl.List(pl.List(pl.Float64)),
                "x": pl.List(pl.List(pl.Float64)),
                "a": pl.List(pl.List(pl.Float64)),
                self.graph_id_column: pl.String,
                self.label_column: pl.Int64,
            }
        )

    def _convert(self):
        # Group and aggregate in one step
        return (
            self.dataset.group_by(Group.BY_FRAME, maintain_order=True)
            .agg(
                pl.map_groups(
                    exprs=self.__exprs_variables,
                    function=self.__compute,
                    return_dtype=self.return_dtypes,
                ).alias("result_dict")
            )
            .unnest("result_dict")
        )

    def to_graph_frames(self) -> List[dict]:
        def process_chunk(chunk: pl.DataFrame) -> List[dict]:
            return [
                {
                    "a": make_sparse(reshape_array(arr=chunk["a"][i])),
                    "x": reshape_array(arr=chunk["x"][i]),
                    "e": reshape_array(arr=chunk["e"][i]),
                    "y": np.asarray([chunk[self.label_column][i]]),
                    "id": chunk[self.graph_id_column][i],
                }
                for i in range(len(chunk))
                if i % self.sample == 0
            ]

        graph_df = self._convert()
        self.graph_frames = [
            graph
            for chunk in graph_df.lazy().collect().iter_slices(self.chunk_size)
            for graph in process_chunk(chunk)
        ]
        return self.graph_frames

    def to_spektral_graphs(self) -> List[Graph]:
        if not self.graph_frames:
            self.to_graph_frames()

        return [
            Graph(
                x=d["x"],
                a=d["a"],
                e=d["e"],
                y=d["y"],
                id=d["id"],
            )
            for d in self.graph_frames
        ]

    def to_pickle(self, file_path: str, verbose: bool = False) -> None:
        """
        We store the 'dict' version of the Graphs to pickle each graph is now a dict with keys x, a, e, and y
        To use for training with Spektral feed the loaded pickle data to CustomDataset(data=pickled_data)
        """
        if not file_path.endswith("pickle.gz"):
            raise ValueError(
                "Only compressed pickle files of type 'some_file_name.pickle.gz' are supported..."
            )

        if not self.graph_frames:
            self.to_graph_frames()

        if verbose:
            print(f"Storing {len(self.graph_frames)} Graphs in {file_path}...")

        import pickle
        import gzip
        from pathlib import Path

        path = Path(file_path)

        directories = path.parent
        directories.mkdir(parents=True, exist_ok=True)

        with gzip.open(file_path, "wb") as file:
            pickle.dump(self.graph_frames, file)

    def to_dict(self):
        def _transform_empty_dicts(d):
            # Function to transform empty dicts to None
            if isinstance(d, dict):
                return {
                    k: _transform_empty_dicts(v) if v != {} else None
                    for k, v in d.items()
                }
            return d

        result = {}
        for attr, value in self.__dict__.items():
            try:
                json.dumps(value)  # Check if value is JSON serializable
                result[attr] = value
            except (TypeError, OverflowError):
                pass  # Skip non-serializable attributes
        return _transform_empty_dicts(result)

    def save(self, file_path: str) -> None:
        """
        Function to save the configuration of the graph converter to a JSON file.
        Args:
            file_path (str): Path to the JSON file.
        """

        package_version = self._get_package_version()
        data_to_save = {
            "package_version": package_version,
            "graph_converter_attributes": self.to_dict(),
            "graph_settings": self.settings.to_dict(),
            # "graph_feature_cols": self.dataset_checkpoint.data.columns
            # + (self.graph_feature_cols or []),
            "dataset_features": self.dataset_checkpoint.get_features(),
            "dataset_cols": self.dataset_checkpoint.data.columns,
            # "graph_feature_cols": self.graph_feature_cols or [],
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"Configuration saved to {file_path}")

    def _get_package_version(self):
        version_file_path = os.path.join(os.path.dirname(__file__), "../../__init__.py")

        if not os.path.exists(version_file_path):
            raise FileNotFoundError(f"__init__.py not found at {version_file_path}")

        with open(version_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("__version__"):
                # Extract the version value
                version = line.split("=")[-1].strip().strip('"')
                return version

        raise ValueError("Version not found in __init__.py")
