import logging
import sys
from copy import deepcopy

import pandas as pd

import warnings

from dataclasses import dataclass, field, asdict

from typing import List, Union, Dict, Literal

from kloppy.domain import (
    TrackingDataset,
    Frame,
    Orientation,
    DatasetTransformer,
    DatasetFlag,
    SecondSpectrumCoordinateSystem,
    MetricPitchDimensions,
)

from spektral.data import Graph

from .exceptions import (
    MissingLabelsError,
    MissingDatasetError,
    IncorrectDatasetTypeError,
    KeyMismatchError,
)

from .graph_settings_pl import GraphSettingsPL
from .dataset import KloppyDataset
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
class SoccerGraphConverterPL(DefaultGraphConverter):
    """
    Converts our dataset TrackingDataset into an internal structure

    Attributes:
        dataset (TrackingDataset): Kloppy TrackingDataset.
        labels (dict): Dict with a key per frame_id, like so {frame_id: True/False/1/0}
        graph_id (str, int): Set a single id for the whole Kloppy dataset.
        graph_ids (dict): Frame level control over graph ids.

        The graph_ids will be used to assign each graph an identifier. This identifier allows us to split the CustomSpektralDataset such that
            all graphs with the same id are either all in the test, train or validation set to avoid leakage. It is recommended to either set graph_id (int, str) as
            a match_id, or pass a dictionary into 'graph_ids' with exactly the same keys as 'labels' for more granualar control over the graph ids.
        The latter can be useful when splitting graphs by possession or sequence id. In this case the dict would be {frame_id: sequence_id/possession_id}.
        Note that sequence_id/possession_id should probably be unique for the whole dataset. Perhaps like so {frame_id: 'match_id-sequence_id'}. Defaults to None.

        infer_ball_ownership (bool):
            Infers 'attacking_team' if no 'ball_owning_team' (Kloppy) or 'attacking_team' (List[Dict]) is provided, by finding player closest to ball using ball xyz.
            Also infers ball_carrier within ball_carrier_threshold
        infer_goalkeepers (bool): set True if no GK label is provider, set False for incomplete (broadcast tracking) data that might not have a GK in every frame
        ball_carrier_threshold (float): The distance threshold to determine the ball carrier. Defaults to 25.0.
        boundary_correction (float): A correction factor for boundary calculations, used to correct out of bounds as a percentages (Used as 1+boundary_correction, ie 0.05). Defaults to None.
        non_potential_receiver_node_value (float): Value between 0 and 1 to assign to the defing team players
    """

    dataset: KloppyDataset = None

    label_col: str = "label"
    graph_id_col: str = "graph_id"

    chunk_size: int = 2_0000

    infer_goalkeepers: bool = True
    infer_ball_ownership: bool = True
    boundary_correction: float = None
    ball_carrier_treshold: float = 25.0

    non_potential_receiver_node_value: float = 0.1

    def __post_init__(self):
        self.pitch_dimensions: MetricPitchDimensions = self.dataset.pitch_dimensions
        self.dataset = self.dataset.data

        self._sport_specific_checks()
        self.settings = self._apply_settings()
        self.dataset = self._apply_filters()

    def _apply_filters(self):
        return self.dataset.with_columns(
            pl.when(
                (pl.col(self.settings._identifier_column) == self.settings.ball_id)
                & (pl.col("v") > self.settings.max_ball_speed)
            )
            .then(self.settings.max_ball_speed)
            .when(
                (pl.col(self.settings._identifier_column) != self.settings.ball_id)
                & (pl.col("v") > self.settings.max_player_speed)
            )
            .then(self.settings.max_player_speed)
            .otherwise(pl.col("v"))
            .alias("v")
        ).with_columns(
            pl.when(
                (pl.col(self.settings._identifier_column) == self.settings.ball_id)
                & (pl.col("a") > self.settings.max_ball_acceleration)
            )
            .then(self.settings.max_ball_acceleration)
            .when(
                (pl.col(self.settings._identifier_column) != self.settings.ball_id)
                & (pl.col("a") > self.settings.max_player_acceleration)
            )
            .then(self.settings.max_player_acceleration)
            .otherwise(pl.col("a"))
            .alias("a")
        )

    def _apply_settings(self):
        return GraphSettingsPL(
            pitch_dimensions=self.pitch_dimensions,
            ball_carrier_treshold=self.ball_carrier_treshold,
            max_player_speed=self.max_player_speed,
            max_ball_speed=self.max_ball_speed,
            max_player_acceleration=self.max_player_acceleration,
            max_ball_acceleration=self.max_ball_acceleration,
            boundary_correction=self.boundary_correction,
            self_loop_ball=self.self_loop_ball,
            adjacency_matrix_connect_type=self.adjacency_matrix_connect_type,
            adjacency_matrix_type=self.adjacency_matrix_type,
            label_type=self.label_type,
            infer_ball_ownership=self.infer_ball_ownership,
            infer_goalkeepers=self.infer_goalkeepers,
            defending_team_node_value=self.defending_team_node_value,
            non_potential_receiver_node_value=self.non_potential_receiver_node_value,
            random_seed=self.random_seed,
            pad=self.pad,
            verbose=self.verbose,
        )

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
        if not isinstance(self.infer_goalkeepers, bool):
            raise Exception("'infer_goalkeepers' should be of type boolean (bool)")

        if not isinstance(self.infer_ball_ownership, bool):
            raise Exception("'infer_ball_ownership' should be of type boolean (bool)")

        if self.boundary_correction and not isinstance(self.boundary_correction, float):
            raise Exception("'boundary_correction' should be of type float")

        if self.ball_carrier_treshold and not isinstance(
            self.ball_carrier_treshold, float
        ):
            raise Exception("'ball_carrier_treshold' should be of type float")

        if self.non_potential_receiver_node_value and not isinstance(
            self.non_potential_receiver_node_value, float
        ):
            raise Exception(
                "'non_potential_receiver_node_value' should be of type float"
            )

    def _convert(self):
        def __compute(args: List[pl.Series]) -> dict:
            x = args[0].to_numpy()
            y = args[1].to_numpy()
            z = args[2].to_numpy()
            v = args[3].to_numpy()
            vx = args[4].to_numpy()
            vy = args[5].to_numpy()
            vz = args[6].to_numpy()
            a = args[7].to_numpy()
            ax = args[8].to_numpy()
            ay = args[9].to_numpy()
            az = args[10].to_numpy()

            team_id = args[6].to_numpy()
            position_name = args[7].to_numpy()
            ball_owning_team_id = args[8].to_numpy()
            graph_id = args[9].to_numpy()
            label = args[10].to_numpy()

            if not np.all(graph_id == graph_id[0]):
                raise Exception(
                    "GraphId selection contains multiple different values. Make sure each GraphId is unique by at least playId and frameId..."
                )

            if not self.prediction and not np.all(label == label[0]):
                raise Exception(
                    "Label selection contains multiple different values for a single selection (group by) of playId and frameId, make sure this is not the case. Each group can only have 1 label."
                )

            ball_carrier_idx = get_ball_carrier_idx(
                x=x,
                y=y,
                z=z,
                team=team_id,
                possession_team=ball_owning_team_id,
                ball_id=self.settings.ball_id,
                threshold=self.settings.ball_carrier_treshold,
            )

            adjacency_matrix = compute_adjacency_matrix_pl(
                team=team_id,
                possession_team=ball_owning_team_id,
                settings=self.settings,
                ball_carrier_idx=ball_carrier_idx,
            )
            edge_features = compute_edge_features_pl(
                adjacency_matrix=adjacency_matrix,
                p3d=np.stack((x, y, z), axis=-1),
                p2d=np.stack((x, y), axis=-1),
                s=v,
                velocity=np.stack((vx, vy), axis=-1),
                team=team_id,
                settings=self.settings,
            )
            node_features = compute_node_features_pl(
                x,
                y,
                s=v,
                velocity=np.stack((vx, vy), axis=-1),
                team=team_id,
                possession_team=ball_owning_team_id,
                is_gk=(position_name == self.settings.goalkeeper_id).astype(int),
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
                self.graph_id_col: graph_id[0],
                self.label_col: label[0],
            }

        result_df = self.dataset.group_by(
            ["game_id", "frame_id"], maintain_order=True
        ).agg(
            pl.map_groups(
                exprs=[
                    "x",
                    "y",
                    "z",
                    "v",
                    "vx",
                    "vy",
                    "vz",
                    "a",
                    "ax",
                    "ay",
                    "az",
                    "team_id",
                    "position_name",
                    "ball_owning_team_id",
                    self.graph_id_col,
                    self.label_col,
                ],
                function=__compute,
            ).alias("result_dict")
        )

        graph_df = result_df.with_columns(
            [
                pl.col("result_dict").struct.field("a").alias("a"),
                pl.col("result_dict").struct.field("e").alias("e"),
                pl.col("result_dict").struct.field("x").alias("x"),
                pl.col("result_dict").struct.field("e_shape_0").alias("e_shape_0"),
                pl.col("result_dict").struct.field("e_shape_1").alias("e_shape_1"),
                pl.col("result_dict").struct.field("x_shape_0").alias("x_shape_0"),
                pl.col("result_dict").struct.field("x_shape_1").alias("x_shape_1"),
                pl.col("result_dict").struct.field("a_shape_0").alias("a_shape_0"),
                pl.col("result_dict").struct.field("a_shape_1").alias("a_shape_1"),
                pl.col("result_dict")
                .struct.field(self.graph_id_col)
                .alias(self.graph_id_col),
                pl.col("result_dict")
                .struct.field(self.label_col)
                .alias(self.label_col),
            ]
        )

        return graph_df.drop("result_dict")

    def to_graph_frames(self) -> List[dict]:
        def __convert_to_graph_data_list(df):
            lazy_df = df.lazy()

            graph_list = []

            for chunk in lazy_df.collect().iter_slices(self.chunk_size):
                chunk_graph_list = [
                    {
                        "a": make_sparse(
                            flatten_to_reshaped_array(
                                arr=chunk["a"][i],
                                s0=chunk["a_shape_0"][i],
                                s1=chunk["a_shape_1"][i],
                            )
                        ),
                        "x": flatten_to_reshaped_array(
                            arr=chunk["x"][i],
                            s0=chunk["x_shape_0"][i],
                            s1=chunk["x_shape_1"][i],
                        ),
                        "e": flatten_to_reshaped_array(
                            arr=chunk["e"][i],
                            s0=chunk["e_shape_0"][i],
                            s1=chunk["e_shape_1"][i],
                        ),
                        "y": np.asarray([chunk[self.label_col][i]]),
                        "id": chunk[self.graph_id_col][i],
                    }
                    for i in range(len(chunk["a"]))
                ]
                graph_list.extend(chunk_graph_list)

            return graph_list

        graph_df = self._convert()
        self.graph_frames = __convert_to_graph_data_list(graph_df)

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
