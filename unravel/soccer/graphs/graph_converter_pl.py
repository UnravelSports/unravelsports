import logging
import sys

from dataclasses import dataclass

from typing import List, Union, Dict, Literal, Any, Optional, Callable

import inspect

from kloppy.domain import (
    MetricPitchDimensions,
)

from spektral.data import Graph

from .graph_settings_pl import GraphSettingsPolars
from ..dataset.kloppy_polars import KloppyPolarsDataset, Column, Group, Constant
from .features import (
    compute_node_features,
    add_global_features,
    compute_adjacency_matrix,
    compute_edge_features,
)

from ...utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


@dataclass(repr=True)
class SoccerGraphConverterPolars(DefaultGraphConverter):
    """
    Converts our dataset TrackingDataset into an internal structure

    Attributes:
        dataset (KloppyPolarsDataset): KloppyPolarsDataset created from a Kloppy dataset.
        chunk_size (int): Determines how many Graphs get processed simultanously.
        non_potential_receiver_node_value (float): Value between 0 and 1 to assign to the defing team players
        global_feature_cols (list[str]): List of columns in the dataset that are Graph level features (e.g. team strength rating, win probabilities etc)
            we want to add to our model. A list of column names corresponding to the Polars dataframe within KloppyPolarsDataset.data
            that are graph level features. They should be joined to the KloppyPolarsDataset.data dataframe such that
            each Group in the group_by has the same value per column. We take the first value of the group, and assign this as a
            "graph level feature" to the ball node.
        global_feature_type: A literal of type "ball" or "all". When set to "ball" the global features will be assigned to only the ball node, if set to "all"
            the they will be assigned to every player and ball in the node features.
        edge_feature_funcs: A list of functions (decorated with @graph_feature(is_custom, feature_type="edge"))
            that take **kwargs as input and return a numpy array (dimensions should match expected (N,N) shape or tuple with multipe (N, N) numpy arrays).
        node_feature_funcs: A list of functions (decorated with @graph_feature(is_custom, feature_type="node"))
            that take **kwargs as input and return a numpy array (dimensions should match expected (N,) shape or (N, k) )
        additional_feature_cols: Column the user has added to the 'KloppyPolarsDataset.data' that are not to be added as global features,
            but can now be accessed by edge_feature_funcs and node_feature_funcs through kwargs.
            (e.g. if the user adds "height" for each player, as a column to the 'KloppyPolarsDataset.data' and
            they want to use it to compute the height difference between all players as an edge feature they would
            pass additional_feature_cols=["height"] and their custom edge feature function can now access kwargs['height'])
    """

    dataset: KloppyPolarsDataset = None

    chunk_size: int = 2_0000
    non_potential_receiver_node_value: float = 0.1

    edge_feature_funcs: List[Callable[[Dict[str, Any]], np.ndarray]] = field(
        repr=False, default_factory=list
    )
    node_feature_funcs: List[Callable[[Dict[str, Any]], np.ndarray]] = field(
        repr=False, default_factory=list
    )

    global_feature_cols: Optional[List[str]] = field(repr=False, default_factory=list)
    global_feature_type: Literal["ball", "all"] = "ball"

    additional_feature_cols: Optional[List[str]] = field(
        repr=False, default_factory=list
    )

    def __post_init__(self):
        if not isinstance(self.dataset, KloppyPolarsDataset):
            raise ValueError("dataset should be of type KloppyPolarsDataset...")

        self.pitch_dimensions: MetricPitchDimensions = (
            self.dataset.settings.pitch_dimensions
        )
        self.label_column: str = (
            self.label_col if self.label_col is not None else self.dataset._label_column
        )
        self.graph_id_column: str = (
            self.graph_id_col
            if self.graph_id_col is not None
            else self.dataset._graph_id_column
        )

        self.dataset = self.dataset.data

        if not self.edge_feature_funcs:
            self.edge_feature_funcs = self.default_edge_feature_funcs

        self._verify_feature_funcs(self.edge_feature_funcs, feature_type="edge")

        if not self.node_feature_funcs:
            self.node_feature_funcs = self.default_node_feature_funcs

        self._verify_feature_funcs(self.node_feature_funcs, feature_type="node")

        self._sport_specific_checks()
        self.settings = self._apply_graph_settings()

        if self.pad:
            self.dataset = self._apply_padding()
        else:
            self.dataset = self._remove_incomplete_frames()

        self._sample()
        self._shuffle()

    def _sample(self):
        if self.sample_rate is None:
            return
        else:
            self.dataset = self.dataset.filter(
                pl.col(Column.FRAME_ID) % (1.0 / self.sample_rate) == 0
            )

    def _verify_feature_funcs(self, funcs, feature_type: Literal["edge", "node"]):
        for i, func in enumerate(funcs):
            # Check if it has the attributes added by the decorator
            if not hasattr(func, "feature_type"):
                func_str = inspect.getsource(func).strip()
                raise Exception(
                    f"Error processing feature function:\n"
                    f"{func.__name__} defined as:\n"
                    f"{func_str}\n\n"
                    "Function is missing the @graph_feature decorator. "
                )

            if func.feature_type != feature_type:
                func_str = inspect.getsource(func).strip()
                raise Exception(
                    f"Error processing feature function:\n"
                    f"{func.__name__} defined as:\n"
                    f"{func_str}\n\n"
                    "Function has an incorrect feature type edge features should be 'edge', node features should be 'node'. "
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
            exprs_variables + self.global_feature_cols + self.additional_feature_cols
        )
        return exprs

    @property
    def default_node_feature_funcs(self) -> list:
        return [
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
        ]

    @property
    def default_edge_feature_funcs(self) -> list:
        return [
            distances_between_players_normed,
            speed_difference_normed,
            angle_between_players_normed,
            velocity_difference_normed,
        ]

    def __add_additional_kwargs(self, d):
        d["ball_id"] = Constant.BALL
        d["possession_team_id"] = d[Column.BALL_OWNING_TEAM_ID][0]
        d["is_gk"] = np.where(
            d[Column.POSITION_NAME] == self.settings.goalkeeper_id, True, False
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
        return d

    def __compute(self, args: List[pl.Series]) -> dict:
        d = {col: args[i].to_numpy() for i, col in enumerate(self.__exprs_variables)}
        d = self.__add_additional_kwargs(d)

        if self.global_feature_cols:
            failed = [
                col
                for col in self.global_feature_cols
                if not np.all(d[col] == d[col][0])
            ]
            if failed:
                raise ValueError(
                    f"""graph_feature_cols contains multiple different values for a group in the groupby ({Group.BY_FRAME}) selection for the columns {failed}. Make sure each group has the same values per individual column."""
                )

        global_features = (
            np.asarray([d[col] for col in self.global_feature_cols]).T[0]
            if self.global_feature_cols
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

        adjacency_matrix = compute_adjacency_matrix(settings=self.settings, **d)
        edge_features = compute_edge_features(
            adjacency_matrix=adjacency_matrix,
            funcs=self.edge_feature_funcs,
            opts=self.feature_opts,
            settings=self.settings,
            **d,
        )

        node_features = compute_node_features(
            funcs=self.node_feature_funcs,
            opts=self.feature_opts,
            settings=self.settings,
            **d,
        )

        if global_features is not None:
            node_features = add_global_features(
                node_features=node_features,
                global_features=global_features,
                global_feature_type=self.global_feature_type,
                **d,
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
