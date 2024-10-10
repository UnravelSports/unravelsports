from dataclasses import dataclass

import polars as pl
import numpy as np

from typing import List

from spektral.data import Graph

from .graph_settings import (
    AmericanFootballGraphSettings,
    AmericanFootballPitchDimensions,
)
from .features import (
    compute_node_features,
    compute_edge_features,
    compute_adjacency_matrix,
)

from ...utils import DefaultGraphConverter, flatten_to_reshaped_array, make_sparse


@dataclass(repr=True)
class AmericanFootballGraphConverter(DefaultGraphConverter):
    def __init__(
        self,
        dataset: pl.DataFrame,
        pitch_dimensions: AmericanFootballPitchDimensions,
        label_col: str = "label",
        graph_id_col: str = "graph_id",
        chunk_size: int = 10_000,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset = dataset
        self.pitch_dimensions = pitch_dimensions
        self.label_col = label_col
        self.graph_id_col = graph_id_col
        self.chunk_size = chunk_size

        self._sport_specific_checks()

        self.settings = self._apply_settings()

    def _sport_specific_checks(self):
        if not self.label_col in self.dataset.columns and not self.prediction:
            raise Exception(
                "Please specify a 'label_col' and add that column to your 'dataset' or set 'prediction=True' if you want to use the converted dataset to make predictions on."
            )

        if not self.graph_id_col in self.dataset.columns:
            raise Exception(
                "Please specify a 'graph_id_col' and add that column to your 'dataset' ..."
            )

    def _apply_settings(self):
        return AmericanFootballGraphSettings(
            pitch_dimensions=self.pitch_dimensions,
            ball_carrier_treshold=self.ball_carrier_treshold,
            max_player_speed=self.max_player_speed,
            max_ball_speed=self.max_ball_speed,
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

    def _convert(self):
        def __compute(args: List[pl.Series]) -> dict:
            x = args[0].to_numpy()
            y = args[1].to_numpy()
            s = args[2].to_numpy()
            a = args[3].to_numpy()
            dis = args[4].to_numpy()
            o = args[5].to_numpy()
            dir = args[6].to_numpy()
            team = args[7].to_numpy()
            official_position = args[8].to_numpy()
            possession_team = args[9].to_numpy()
            graph_id = args[10].to_numpy()
            label = args[11].to_numpy()

            if not np.all(graph_id == graph_id[0]):
                raise Exception(
                    "GraphId selection contains multiple different values. Make sure each GraphId is unique by at least playId and frameId..."
                )

            if not np.all(label == label[0]):
                raise Exception(
                    "Label selection contains multiple different values for a single selection (group by) of playId and frameId, make sure this is not the case. Each group can only have 1 label."
                )

            adjacency_matrix = compute_adjacency_matrix(
                team=team, possession_team=possession_team, settings=self.settings
            )

            edge_features = compute_edge_features(
                adjacency_matrix=adjacency_matrix,
                p=np.stack((x, y), axis=-1),
                s=s,
                a=a,
                dir=dir,
                o=o,  # Shape will be (N, 2)
                settings=self.settings,
            )

            node_features = compute_node_features(
                x,
                y,
                s=s,
                a=a,
                dir=dir,
                o=o,
                team=team,
                official_position=official_position,
                possession_team=possession_team,
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
            ["playId", "frameId"], maintain_order=True
        ).agg(
            pl.map_groups(
                exprs=[
                    "x",
                    "y",
                    "s",
                    "a",
                    "dis",
                    "o",
                    "dir",
                    "team",
                    "officialPosition",
                    "possessionTeam",
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
