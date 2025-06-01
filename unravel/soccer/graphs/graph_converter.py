import logging
import sys

from dataclasses import dataclass

from typing import List, Union, Dict, Literal, Any, Optional, Callable

import inspect

import pathlib

from kloppy.domain import MetricPitchDimensions, Orientation

from spektral.data import Graph

from .graph_settings import GraphSettingsPolars
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
class SoccerGraphConverter(DefaultGraphConverter):
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

    _edge_feature_dims: Dict[str, int] = field(
        repr=False, default_factory=dict, init=False
    )
    _node_feature_dims: Dict[str, int] = field(
        repr=False, default_factory=dict, init=False
    )

    def __post_init__(self):
        if not isinstance(self.dataset, KloppyPolarsDataset):
            raise ValueError("dataset should be of type KloppyPolarsDataset...")

        self.pitch_dimensions: MetricPitchDimensions = (
            self.dataset.settings.pitch_dimensions
        )
        self._kloppy_settings = self.dataset.settings

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

    @staticmethod
    def _sort(df):
        sort_expr = (pl.col(Column.TEAM_ID) == Constant.BALL).cast(int) * 2 - (
            (pl.col(Column.BALL_OWNING_TEAM_ID) == pl.col(Column.TEAM_ID))
            & (pl.col(Column.TEAM_ID) != Constant.BALL)
        ).cast(int)

        df = df.sort([*Group.BY_FRAME, sort_expr, pl.col(Column.OBJECT_ID)])
        return df

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
            self.label_column,
            self.graph_id_column,
        ]
        empty_columns = [
            Column.POSITION_NAME,
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

        user_defined_columns = [
            x
            for x in df.columns
            if x
            not in keep_columns
            + group_by_columns
            + empty_columns
            + self.global_feature_cols
        ]

        counts = df.group_by(group_by_columns).agg(
            pl.len().alias("count"),
            *[
                pl.first(col).alias(col)
                for col in keep_columns + self.global_feature_cols
            ],
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

        padding_rows = []
        # This is where we pad players (missing balls get skipped because of 'target_length')
        for row in groups_to_pad.iter_rows(named=True):
            base_row = {
                col: row[col]
                for col in keep_columns + group_by_columns + self.global_feature_cols
            }
            padding_rows.extend([base_row] * row["repeats"])

        # Now check if there are frames without ball rows
        # Get all unique frames
        all_frames = df.select(
            [
                Column.GAME_ID,
                Column.PERIOD_ID,
                Column.FRAME_ID,
                Column.BALL_OWNING_TEAM_ID,
            ]
            + keep_columns
            + self.global_feature_cols
        ).unique()

        # Get frames that have ball rows
        frames_with_ball = (
            df.filter(pl.col(Column.TEAM_ID) == Constant.BALL)
            .select([Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID])
            .unique()
        )

        # Find frames missing ball rows
        frames_missing_ball = all_frames.join(
            frames_with_ball,
            on=[Column.GAME_ID, Column.PERIOD_ID, Column.FRAME_ID],
            how="anti",
        )

        # Create a dataframe of ball rows to add with appropriate columns
        if frames_missing_ball.height > 0:
            # Create base rows for missing balls
            ball_rows_to_add = frames_missing_ball.with_columns(
                [
                    pl.lit(Constant.BALL).alias(Column.TEAM_ID),
                    pl.lit(Constant.BALL).alias(Column.POSITION_NAME),
                ]
            )

            # Add to padding rows using same pattern as for players
            for row in ball_rows_to_add.iter_rows(named=True):
                base_row = {
                    col: row[col]
                    for col in keep_columns
                    + group_by_columns
                    + [Column.POSITION_NAME]
                    + self.global_feature_cols
                    if col in row
                }
                padding_rows.append(base_row)

        if len(padding_rows) == 0:
            return df

        padding_df = pl.DataFrame(padding_rows)

        schema = df.schema

        padding_df = padding_df.with_columns(
            [create_default_expression(col, schema[col]) for col in empty_columns]
            + [
                pl.lit(None).cast(schema[col]).alias(col)
                for col in user_defined_columns
            ]
        )
        padding_df = padding_df.with_columns(
            [pl.col(col).cast(df.schema[col]).alias(col) for col in group_by_columns]
        )

        padding_df = padding_df.join(
            (
                df.unique(group_by_columns).select(
                    group_by_columns + self.global_feature_cols
                )
            ),
            on=group_by_columns,
            how="left",
        )

        padding_df = padding_df.with_columns(
            [
                pl.col(col_name).cast(df.schema[col_name]).alias(col_name)
                for col_name in df.columns
            ]
        ).select(df.columns)

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
            home_team_id=str(self._kloppy_settings.home_team_id),
            away_team_id=str(self._kloppy_settings.away_team_id),
            orientation=self._kloppy_settings.orientation,
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
    def _exprs_variables(self):
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
        d["position"] = np.nan_to_num(
            np.stack((d[Column.X], d[Column.Y], d[Column.Z]), axis=-1),
            nan=1e-10,
            posinf=1e3,
            neginf=-1e3,
        )
        d["velocity"] = np.nan_to_num(
            np.stack((d[Column.VX], d[Column.VY], d[Column.VZ]), axis=-1),
            nan=1e-10,
            posinf=1e3,
            neginf=-1e3,
        )

        if len(np.where(d["team_id"] == d["ball_id"])[0]) >= 1:
            ball_index = np.where(d["team_id"] == d["ball_id"])[0]
            ball_position = d["position"][ball_index][0]
        else:
            ball_position = np.asarray([0.0, 0.0, 0.0])
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

    def _compute(self, args: List[pl.Series]) -> dict:
        frame_data: dict = {
            col: args[i].to_numpy() for i, col in enumerate(self._exprs_variables)
        }
        frame_data = self.__add_additional_kwargs(frame_data)
        frame_id = args[-1][0]

        if not np.all(
            frame_data[self.graph_id_column] == frame_data[self.graph_id_column][0]
        ):
            raise ValueError(
                "graph_id selection contains multiple different values. Make sure each graph_id is unique by at least game_id and frame_id..."
            )

        if not self.prediction and not np.all(
            frame_data[self.label_column] == frame_data[self.label_column][0]
        ):
            raise ValueError(
                """Label selection contains multiple different values for a single selection (group by) of game_id and frame_id, 
                make sure this is not the case. Each group can only have 1 label."""
            )

        adjacency_matrix = compute_adjacency_matrix(
            settings=self.settings, **frame_data
        )
        edge_features, self._edge_feature_dims = compute_edge_features(
            adjacency_matrix=adjacency_matrix,
            funcs=self.edge_feature_funcs,
            opts=self.feature_opts,
            settings=self.settings,
            **frame_data,
        )

        node_features, self._node_feature_dims = compute_node_features(
            funcs=self.node_feature_funcs,
            opts=self.feature_opts,
            settings=self.settings,
            **frame_data,
        )

        if self.global_feature_cols:
            failed = [
                col
                for col in self.global_feature_cols
                if not np.all(frame_data[col] == frame_data[col][0])
            ]
            if failed:
                raise ValueError(
                    f"""graph_feature_cols contains multiple different values for a group in the groupby ({Group.BY_FRAME}) selection for the columns {failed}. Make sure each group has the same values per individual column."""
                )

            global_features = (
                np.asarray([frame_data[col] for col in self.global_feature_cols]).T[0]
                if self.global_feature_cols
                else None
            )
            for col in self.global_feature_cols:
                self._node_feature_dims[col] = 1

            node_features = add_global_features(
                node_features=node_features,
                global_features=global_features,
                global_feature_type=self.global_feature_type,
                **frame_data,
            )
        return {
            "e": pl.Series(
                [edge_features.tolist()], dtype=pl.List(pl.List(pl.Float64))
            ),
            "x": pl.Series(
                [node_features.tolist()], dtype=pl.List(pl.List(pl.Float64))
            ),
            "a": pl.Series(
                [adjacency_matrix.tolist()], dtype=pl.List(pl.List(pl.Int32))
            ),
            "e_shape_0": edge_features.shape[0],
            "e_shape_1": edge_features.shape[1],
            "x_shape_0": node_features.shape[0],
            "x_shape_1": node_features.shape[1],
            "a_shape_0": adjacency_matrix.shape[0],
            "a_shape_1": adjacency_matrix.shape[1],
            self.graph_id_column: frame_data[self.graph_id_column][0],
            self.label_column: frame_data[self.label_column][0],
            "frame_id": frame_id,
        }

    def _convert(self):
        # Group and aggregate in one step
        return (
            self.dataset.group_by(Group.BY_FRAME, maintain_order=True)
            .agg(
                pl.map_groups(
                    exprs=self._exprs_variables + [Column.FRAME_ID],
                    function=self._compute,
                    return_dtype=self.return_dtypes,
                ).alias("result_dict")
            )
            .with_columns(
                [
                    *[
                        pl.col("result_dict").struct.field(f).alias(f)
                        for f in [
                            "a",
                            "e",
                            "x",
                            self.graph_id_column,
                            self.label_column,
                            "frame_id",
                        ]
                    ],
                    *[
                        pl.col("result_dict")
                        .struct.field(f"{m}_shape_{i}")
                        .alias(f"{m}_shape_{i}")
                        for m in ["a", "e", "x"]
                        for i in [0, 1]
                    ],
                ]
            )
            .drop("result_dict")
        )

    def plot(
        self,
        file_path: str,
        fps: int = None,
        timestamp: pl.duration = None,
        end_timestamp: pl.duration = None,
        period_id: int = None,
        team_color_a: str = "#CD0E61",
        team_color_b: str = "#0066CC",
        ball_color: str = "black",
        sort: bool = True,
        color_by: Literal["ball_owning", "static_home_away"] = "ball_owning",
    ):
        """
        Plot tracking data as a static image or video file.

        This method visualizes tracking data for players and the ball. It can generate either:
        - A single PNG image (if either fps or end_timestamp is None, or both are None)
        - An MP4 video (if both fps and end_timestamp are provided)

        Parameters
        ----------
        file_path : str
            The output path where the PNG or MP4 file will be saved
        fps : int, optional
            Frames per second for video output. If None, a static image is generated
        timestamp : pl.duration, optional
            The starting timestamp to plot. If None, starts from the beginning of available data
        end_timestamp : pl.duration, optional
            The ending timestamp for video output. If None, a static image is generated
        period_id : int, optional
            ID of the match period to visualize. If None, all periods are included
        team_color_a : str, default "#CD0E61"
            Hex color code for Team A visualization
        team_color_b : str, default "#0066CC"
            Hex color code for Team B visualization
        ball_color : str, default "black"
            Color for ball visualization
        color_by : Literal["ball_owning", "static_home_away"], default "ball_owning"
            Method for coloring the teams:
            - "ball_owning": Colors teams based on ball possession
            - "static_home_away": Uses static colors for home and away teams

        Returns
        -------
        None
            The function saves the output file to the specified file_path but doesn't return any value

        Notes
        -----
        Output file type is determined by parameters:
        - PNG: Generated when either fps or end_timestamp is None, or both are None
        - MP4: Generated when both fps and end_timestamp are provided

        Raises
        ------
        ValueError
            If file extension doesn't match the parameters provided (e.g., .mp4 extension
            but missing fps or end_timestamp, or .png extension with both fps and end_timestamp)
        """
        try:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            raise ImportError(
                "Seems like you don't have matplotlib installed. Please"
                " install it using: pip install matplotlib"
            )

        if (fps is None and end_timestamp is not None) or (
            fps is not None and end_timestamp is None
        ):
            raise ValueError(
                "Both 'fps' and 'end_timestamp' must be provided together to generate a video. "
            )

        # Determine the output type based on parameters
        generate_video = fps is not None and end_timestamp is not None

        # Get file extension if it exists
        path = pathlib.Path(file_path)
        file_extension = path.suffix.lower() if path.suffix else ""

        # If no extension, add the appropriate one based on parameters
        if not file_extension:
            suffix = ".mp4" if generate_video else ".png"
            file_path = str(path.with_suffix(suffix))

        # Otherwise, validate that the extension matches the parameters
        else:
            if generate_video and file_extension != ".mp4":
                raise ValueError(
                    f"Parameters fps and end_timestamp indicate video output, "
                    f"but file extension is '{file_extension}'. Use '.mp4' extension for video output."
                )
            elif not generate_video and file_extension == ".mp4":
                raise ValueError(
                    "To generate an MP4 video, both 'fps' and 'end_timestamp' must be provided. "
                    "For static image output, use a '.png' extension."
                )
            elif not generate_video and file_extension != ".png":
                raise ValueError(
                    f"For static image output, use '.png' extension instead of '{file_extension}'."
                )

        self._team_color_a = team_color_a
        self._team_color_b = team_color_b
        self._ball_color = ball_color
        self._color_by = color_by

        if period_id is not None and not isinstance(period_id, int):
            raise TypeError("period_id should be of type integer")

        if all(x is None for x in [timestamp, end_timestamp, period_id]):
            # No filters specified, use the entire dataset
            df = self.dataset
        elif timestamp is not None and period_id is not None:
            if end_timestamp is not None:
                # Both timestamp and end_timestamp provided - filter for a range
                df = self.dataset.filter(
                    (pl.col(Column.TIMESTAMP).is_between(timestamp, end_timestamp))
                    & (pl.col(Column.PERIOD_ID) == period_id)
                )
            else:
                # Only timestamp provided (no end_timestamp) - filter for specific timestamp
                df = self.dataset.filter(
                    (pl.col(Column.TIMESTAMP) == timestamp)
                    & (pl.col(Column.PERIOD_ID) == period_id)
                )
                # Handle the case where a single timestamp has multiple frame_ids
                df = (
                    df.with_columns(
                        pl.col(Column.FRAME_ID)
                        .rank(method="min")
                        .over(Column.TIMESTAMP)
                        .alias("frame_rank")
                    )
                    # Keep only rows where the frame has rank = 1 (first frame for each timestamp)
                    .filter(pl.col("frame_rank") == 1).drop("frame_rank")
                )
        else:
            raise ValueError(
                "Please specify both timestamp and period_id, or specify all of timestamp, end_timestamp, and period_id, or none of them."
            )

        if df.is_empty():
            raise ValueError("Selection is empty, please try different timestamp(s)")

        def plot_graph():
            import matplotlib.pyplot as plt

            # Plot node features in top-left
            ax1 = self._fig.add_subplot(self._gs[0, 0])
            ax1.imshow(self._graph.x, aspect="auto", cmap="YlOrRd")
            ax1.set_xlabel(f"Node Features {self._graph.x.shape}")

            # Set y labels to integers
            num_rows = self._graph.x.shape[0]
            ax1.set_yticks(range(num_rows))
            ax1.set_yticklabels([str(i) for i in range(num_rows)])

            node_feature_yticklabels = feature_ticklabels(self._node_feature_dims)
            ax1.xaxis.set_ticks_position("top")
            ax1.set_xticks(range(len(node_feature_yticklabels)))
            ax1.set_xticklabels(node_feature_yticklabels, rotation=45, ha="left")

            # Plot ajacency matrix in bottom-left
            ax2 = self._fig.add_subplot(self._gs[1, 0])
            ax2.imshow(self._graph.a.toarray(), aspect="auto", cmap="YlOrRd")
            ax2.set_xlabel(f"Adjacency Matrix {self._graph.a.shape}")

            # Set both x and y labels to integers
            num_rows_a = self._graph.a.toarray().shape[0]
            num_cols_a = self._graph.a.toarray().shape[1]

            ax2.set_yticks(range(num_rows_a))
            ax2.set_yticklabels([str(i) for i in range(num_rows_a)])
            ax2.xaxis.set_ticks_position("top")
            ax2.set_xticks(range(num_cols_a))
            ax2.set_xticklabels([str(i) for i in range(num_cols_a)])

            # Plot Edge Features on the right (spanning both rows)
            ax3 = self._fig.add_subplot(self._gs[:, 1])

            _, size_a = non_zeros(self._graph.a.toarray()[0 : self._ball_carrier_idx])
            ball_carrier_edge_idx, num_rows_e = non_zeros(
                np.asarray(
                    [list(x) for x in self._graph.a.toarray()][self._ball_carrier_idx]
                )
            )

            im3 = ax3.imshow(
                self._graph.e[size_a : num_rows_e + size_a, :],
                aspect="auto",
                cmap="YlOrRd",
            )

            ax3.set_yticks(range(num_rows_e))
            ax3.set_yticklabels(list(ball_carrier_edge_idx[0]), fontsize=18)
            ax3.set_xlabel(f"Edge Features {self._graph.e.shape}")

            labels = ax3.get_yticklabels()
            if self._ball_carrier_idx in ball_carrier_edge_idx[0]:
                idx_position = list(ball_carrier_edge_idx[0]).index(
                    self._ball_carrier_idx
                )
                # Modify just that specific label
                labels[idx_position].set_color(self._ball_carrier_color)
                labels[idx_position].set_fontweight("bold")
                # Set the modified labels back
                ax3.set_yticklabels(labels)

            # Set x labels to edge function names at the top, rotated 45 degrees
            edge_feature_xticklabels = feature_ticklabels(self._edge_feature_dims)
            ax3.xaxis.set_ticks_position("top")
            ax3.set_xticks(range(len(edge_feature_xticklabels)))
            ax3.set_xticklabels(edge_feature_xticklabels, rotation=45, ha="left")

            plt.colorbar(im3, ax=ax3, fraction=0.1, pad=0.2)

        def plot_vertical_pitch(frame_data: pl.DataFrame):
            try:
                from mplsoccer import VerticalPitch
            except ImportError:
                raise ImportError(
                    "Seems like you don't have mplsoccer installed. Please"
                    " install it using: pip install mplsoccer"
                )

            ax4 = self._fig.add_subplot(self._gs[:, 2])
            pitch = VerticalPitch(
                pitch_type="secondspectrum",
                pitch_length=self.pitch_dimensions.pitch_length,
                pitch_width=self.pitch_dimensions.pitch_width,
                pitch_color="#ffffff",
                pad_top=-0.05,
            )
            pitch.draw(ax=ax4)
            player_and_ball(frame_data=frame_data, ax=ax4)
            direction_of_play_arrow(ax=ax4)

        def feature_ticklabels(feature_dims):
            _feature_ticklabels = []
            for key, value in feature_dims.items():
                if value == 1:
                    _feature_ticklabels.append(key)
                else:
                    _feature_ticklabels.extend([key] + [None] * (value - 1))
            return _feature_ticklabels

        def direction_of_play_arrow(ax):
            arrow_x = -30
            arrow_y = -7.5
            arrow_dx = 0
            arrow_dy = 15

            if self.settings.orientation == Orientation.STATIC_HOME_AWAY:
                if self._ball_owning_team_id != self.settings.home_team_id:
                    arrow_y = arrow_y * -1
                    arrow_dy = arrow_dy * -1
            elif self.settings.orientation == Orientation.BALL_OWNING_TEAM:
                pass
            else:
                raise ValueError(f"Unsupported orientation {self.settings.orientation}")

            # Create the arrow to indicate direction of play
            ax.arrow(
                arrow_x,
                arrow_y,
                arrow_dx,
                arrow_dy,
                head_width=3,
                head_length=2,
                fc="#c2c2c2",
                ec="#c2c2c2",
                width=0.5,
                length_includes_head=True,
                zorder=1,
            )

        def player_and_ball(frame_data, ax):
            if self._color_by == "ball_owning":
                team_id = self._ball_owning_team_id
            elif self._color_by == "static_home_away":
                team_id = self.settings.home_team_id
            else:
                raise ValueError(f"Unsupported color_by {self._color_by}")

            self._ball_carrier_color = None

            for i, r in enumerate(frame_data.iter_rows(named=True)):
                v, vy, vx, y, x = (
                    r[Column.SPEED],
                    r[Column.VX],
                    r[Column.VY],
                    r[Column.X],
                    r[Column.Y],
                )
                is_ball = True if r[Column.TEAM_ID] == self.settings.ball_id else False

                if not is_ball:
                    if team_id is None:
                        team_id = r[Column.TEAM_ID]

                    color = (
                        self._team_color_a
                        if r[Column.TEAM_ID] == team_id
                        else self._team_color_b
                    )

                    if r[Column.IS_BALL_CARRIER] == True:
                        self._ball_carrier_color = color

                    ax.scatter(x, y, color=color, s=450)

                    if v > 1.0:
                        ax.annotate(
                            "",
                            xy=(x + vx, y + vy),
                            xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color=color, lw=3),
                        )

                else:
                    ax.scatter(x, y, color=self._ball_color, s=250, zorder=10)
                # # Text with white border
                text = ax.text(
                    x + (-1.2 if is_ball else 0.0),
                    y + (-1.2 if is_ball else 0.0),
                    i,
                    color=self._ball_color if is_ball else color,
                    fontsize=12,
                    ha="center",
                    va="center",
                    zorder=15 if is_ball else 5,
                )

                import matplotlib.patheffects as path_effects

                text.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=6, foreground="white"),
                        path_effects.Normal(),
                    ]
                )
                ax.set_xlabel(f"Label: {frame_data['label'][0]}", fontsize=22)
                ax.set_title(self._gameclock, fontsize=22)

        def frame_plot(self, frame_data):
            def timestamp_to_gameclock(timestamp, period_id):
                total_seconds = timestamp.total_seconds()

                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                milliseconds = int((total_seconds % 1) * 1000)

                return f"[{period_id}] - {minutes}:{seconds:02d}:{milliseconds:03d}"

            self._gs = GridSpec(
                2,
                3,
                width_ratios=[2, 1, 3],
                height_ratios=[1, 1],
                wspace=0.1,
                hspace=0.06,
                left=0.05,
                right=1.0,
                bottom=0.05,
            )

            # Process the current frame
            features = self._compute([frame_data[col] for col in self._exprs_variables])
            a = make_sparse(
                reshape_from_size(
                    features["a"], features["a_shape_0"], features["a_shape_1"]
                )
            )
            x = reshape_from_size(
                features["x"], features["x_shape_0"], features["x_shape_1"]
            )
            e = reshape_from_size(
                features["e"], features["e_shape_0"], features["e_shape_1"]
            )
            y = np.asarray([features[self.label_column]])
            frame_id = features["frame_id"]

            self._graph = Graph(
                a=a,
                x=x,
                e=e,
                y=y,
            )

            self._ball_carrier_idx = np.where(
                frame_data[Column.IS_BALL_CARRIER] == True
            )[0][0]
            self._ball_owning_team_id = list(frame_data[Column.BALL_OWNING_TEAM_ID])[0]
            self._gameclock = timestamp_to_gameclock(
                timestamp=list(frame_data["timestamp"])[0],
                period_id=list(frame_data["period_id"])[0],
            )

            plot_vertical_pitch(frame_data)
            plot_graph()

            plt.tight_layout()

        self._fig = plt.figure(figsize=(25, 18))
        self._fig.subplots_adjust(left=0.06, right=1.0, bottom=0.05)

        if sort:
            df = self._sort(df)

        if generate_video:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1500)

            with writer.saving(self._fig, file_path, dpi=300):
                for group_id, frame_data in df.group_by(
                    Group.BY_FRAME, maintain_order=True
                ):
                    self._fig.clear()
                    frame_plot(self, frame_data)
                    writer.grab_frame()

        else:
            frame_plot(self, frame_data=df)
            plt.savefig(file_path, dpi=300)
