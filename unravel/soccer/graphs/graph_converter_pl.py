import logging
import sys

from dataclasses import dataclass

from typing import List, Union, Dict, Literal, Any

from kloppy.domain import (
    MetricPitchDimensions,
)

from spektral.data import Graph

from .graph_settings_pl import GraphSettingsPolars
from .dataset import KloppyPolarsDataset, Column, Group, Constant
from .features import (
    compute_node_features_pl,
    compute_adjacency_matrix_pl,
    compute_edge_features_pl,
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
        dataset (TrackingDataset): Kloppy TrackingDataset.
        chunk_size (int): Determines how many Graphs get processed simultanously.
        non_potential_receiver_node_value (float): Value between 0 and 1 to assign to the defing team players
    """

    dataset: KloppyPolarsDataset = None

    chunk_size: int = 2_0000
    non_potential_receiver_node_value: float = 0.1

    def __post_init__(self):
        if not isinstance(self.dataset, KloppyPolarsDataset):
            raise ValueError("dataset should be of type KloppyPolarsDataset...")

        self.pitch_dimensions: MetricPitchDimensions = self.dataset.pitch_dimensions
        self.label_column: str = (
            self.label_col if self.label_col is not None else self.dataset._label_column
        )
        self.graph_id_column: str = (
            self.graph_id_col
            if self.graph_id_col is not None
            else self.dataset._graph_id_column
        )

        self.dataset = self.dataset.data

        self._sport_specific_checks()
        self.settings = self._apply_settings()
        self.dataset = self._apply_filters()

        if self.pad:
            self.dataset = self._apply_padding()

        self._shuffle()

    def _shuffle(self):
        if isinstance(self.settings.random_seed, int):
            self.dataset = self.dataset.sample(
                fraction=1.0, seed=self.settings.random_seed
            )
        elif self.settings.random_seed == True:
            self.dataset = self.dataset.sample(fraction=1.0)
        else:
            pass

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
        if dropped_frames > 0:
            import warnings

            warnings.warn(
                f"""Setting pad=True drops frames that do not have at least 1 object for the attacking team, defending team or ball.
                This operation dropped {dropped_frames} incomplete frames out of {total_frames} total frames ({(dropped_frames/total_frames)*100:.2f}%)
                """
            )

        return result.join(frame_completeness, on=Group.BY_FRAME, how="inner")

    def _apply_filters(self):
        return self.dataset.with_columns(
            pl.when(
                (pl.col(Column.OBJECT_ID) == Constant.BALL)
                & (pl.col(Column.SPEED) > self.settings.max_ball_speed)
            )
            .then(self.settings.max_ball_speed)
            .when(
                (pl.col(Column.OBJECT_ID) != Constant.BALL)
                & (pl.col(Column.SPEED) > self.settings.max_player_speed)
            )
            .then(self.settings.max_player_speed)
            .otherwise(pl.col(Column.SPEED))
            .alias(Column.SPEED)
        ).with_columns(
            pl.when(
                (pl.col(Column.OBJECT_ID) == Constant.BALL)
                & (pl.col(Column.ACCELERATION) > self.settings.max_ball_acceleration)
            )
            .then(self.settings.max_ball_acceleration)
            .when(
                (pl.col(Column.OBJECT_ID) != Constant.BALL)
                & (pl.col(Column.ACCELERATION) > self.settings.max_player_acceleration)
            )
            .then(self.settings.max_player_acceleration)
            .otherwise(pl.col(Column.ACCELERATION))
            .alias(Column.ACCELERATION)
        )

    def _apply_settings(self):
        return GraphSettingsPolars(
            pitch_dimensions=self.pitch_dimensions,
            max_player_speed=self.max_player_speed,
            max_ball_speed=self.max_ball_speed,
            max_player_acceleration=self.max_player_acceleration,
            max_ball_acceleration=self.max_ball_acceleration,
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
        return [
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

    def __compute(self, args: List[pl.Series]) -> dict:
        d = {col: args[i].to_numpy() for i, col in enumerate(self.__exprs_variables)}

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
            settings=self.settings,
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
            self.graph_id_column: d[self.graph_id_column][0],
            self.label_column: d[self.label_column][0],
        }

    def _convert(self):
        # Group and aggregate in one step
        return (
            self.dataset.group_by(Group.BY_FRAME, maintain_order=True)
            .agg(
                pl.map_groups(
                    exprs=self.__exprs_variables, function=self.__compute
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

    def to_graph_frames(self) -> List[dict]:
        def process_chunk(chunk: pl.DataFrame) -> List[dict]:
            return [
                {
                    "a": make_sparse(
                        reshape_array(
                            chunk["a"][i], chunk["a_shape_0"][i], chunk["a_shape_1"][i]
                        )
                    ),
                    "x": reshape_array(
                        chunk["x"][i], chunk["x_shape_0"][i], chunk["x_shape_1"][i]
                    ),
                    "e": reshape_array(
                        chunk["e"][i], chunk["e_shape_0"][i], chunk["e_shape_1"][i]
                    ),
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

    def to_pickle(self, file_path: str) -> None:
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

        import pickle
        import gzip
        from pathlib import Path

        path = Path(file_path)

        directories = path.parent
        directories.mkdir(parents=True, exist_ok=True)

        with gzip.open(file_path, "wb") as file:
            pickle.dump(self.graph_frames, file)
