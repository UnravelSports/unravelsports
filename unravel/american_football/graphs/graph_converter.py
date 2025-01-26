from dataclasses import dataclass

import polars as pl
import numpy as np

from typing import List

from spektral.data import Graph

from .dataset import BigDataBowlDataset, Group, Column, Constant

from .graph_settings import (
    AmericanFootballGraphSettings,
    AmericanFootballPitchDimensions,
)
from .features import (
    compute_node_features,
    compute_edge_features,
    compute_adjacency_matrix,
)

from ...utils import DefaultGraphConverter, reshape_array, make_sparse


@dataclass(repr=True)
class AmericanFootballGraphConverter(DefaultGraphConverter):
    """
    Converts our dataset TrackingDataset into an internal structure

    Attributes:
        dataset (TrackingDataset): Kloppy TrackingDataset.
        label_col (str): Column name that contains labels in the dataset.data Polars dataframe
        graph_id_col (str): Column name that contains graph ids in the dataset.data Polars dataframe

        chunk_size (int): Used to batch convert Polars into Graphs
        attacking_non_qb_node_value (float): Value between 0 and 1 to assign any attacking team player who is not the QB
    """

    def __init__(
        self,
        dataset: BigDataBowlDataset,
        chunk_size: int = 2_000,
        attacking_non_qb_node_value: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(dataset, BigDataBowlDataset):
            raise Exception("'dataset' should be an instance of BigDataBowlDataset")

        self.label_col = dataset._label_column
        self.graph_id_col = dataset._graph_id_column

        self.dataset: pl.DataFrame = dataset.data
        self.pitch_dimensions: AmericanFootballPitchDimensions = (
            dataset.pitch_dimensions
        )
        self.chunk_size = chunk_size
        self.attacking_non_qb_node_value = attacking_non_qb_node_value

        self._sport_specific_checks()

        self.settings = self._apply_settings()

    def _sport_specific_checks(self):

        if not isinstance(self.label_col, str):
            raise Exception("'label_col' should be of type string (str)")

        if not isinstance(self.graph_id_col, str):
            raise Exception("'graph_id_col' should be of type string (str)")

        if not isinstance(self.chunk_size, int):
            raise Exception("chunk_size should be of type integer (int)")

        if not self.label_col in self.dataset.columns and not self.prediction:
            raise Exception(
                "Please specify a 'label_col' and add that column to your 'dataset' or set 'prediction=True' if you want to use the converted dataset to make predictions on."
            )

        if not self.graph_id_col in self.dataset.columns:
            raise Exception(
                "Please specify a 'graph_id_col' and add that column to your 'dataset' ..."
            )

        # Parameter Checks
        if not isinstance(self.attacking_non_qb_node_value, (int, float)):
            raise Exception(
                "'attacking_non_qb_node_value' should be of type float or integer (int)"
            )

    def _apply_settings(self):
        return AmericanFootballGraphSettings(
            pitch_dimensions=self.pitch_dimensions,
            max_player_speed=self.max_player_speed,
            max_ball_speed=self.max_ball_speed,
            max_ball_acceleration=self.max_ball_acceleration,
            max_player_acceleration=self.max_player_acceleration,
            self_loop_ball=self.self_loop_ball,
            adjacency_matrix_connect_type=self.adjacency_matrix_connect_type,
            adjacency_matrix_type=self.adjacency_matrix_type,
            label_type=self.label_type,
            defending_team_node_value=self.defending_team_node_value,
            attacking_non_qb_node_value=self.attacking_non_qb_node_value,
            random_seed=self.random_seed,
            pad=self.pad,
            verbose=self.verbose,
        )

    @property
    def __exprs_variables(self):
        return [
            Column.X,
            Column.Y,
            Column.SPEED,
            Column.ACCELERATION,
            Column.ORIENTATION,
            Column.DIRECTION,
            Column.TEAM,
            Column.OFFICIAL_POSITION,
            Column.POSSESSION_TEAM,
            Column.HEIGHT_CM,
            Column.WEIGHT_KG,
            self.graph_id_col,
            self.label_col,
        ]

    def __compute(self, args: List[pl.Series]) -> dict:
        d = {col: args[i].to_numpy() for i, col in enumerate(self.__exprs_variables)}

        if not np.all(d[self.graph_id_col] == d[self.graph_id_col][0]):
            raise Exception(
                "GraphId selection contains multiple different values. Make sure each graph_id is unique by at least game_id and frame_id..."
            )

        if not self.prediction and not np.all(
            d[self.label_col] == d[self.label_col][0]
        ):
            raise Exception(
                """Label selection contains multiple different values for a single selection (group by) of game_id and frame_id, 
                make sure this is not the case. Each group can only have 1 label."""
            )

        adjacency_matrix = compute_adjacency_matrix(
            team=d[Column.TEAM],
            possession_team=d[Column.POSSESSION_TEAM],
            settings=self.settings,
        )
        edge_features = compute_edge_features(
            adjacency_matrix=adjacency_matrix,
            p=np.stack((d[Column.X], d[Column.Y]), axis=-1),
            s=d[Column.SPEED],
            a=d[Column.ACCELERATION],
            dir=d[Column.DIRECTION],
            o=d[Column.ORIENTATION],
            team=d[Column.TEAM],
            settings=self.settings,
        )
        node_features = compute_node_features(
            x=d[Column.X],
            y=d[Column.Y],
            s=d[Column.SPEED],
            a=d[Column.ACCELERATION],
            dir=d[Column.DIRECTION],
            o=d[Column.ORIENTATION],
            team=d[Column.TEAM],
            official_position=d[Column.OFFICIAL_POSITION],
            possession_team=d[Column.POSSESSION_TEAM],
            height=d[Column.HEIGHT_CM],
            weight=d[Column.WEIGHT_KG],
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
            self.graph_id_col: d[self.graph_id_col][0],
            self.label_col: d[self.label_col][0],
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
                        for f in ["a", "e", "x", self.graph_id_col, self.label_col]
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
                    "y": np.asarray([chunk[self.label_col][i]]),
                    "id": chunk[self.graph_id_col][i],
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
