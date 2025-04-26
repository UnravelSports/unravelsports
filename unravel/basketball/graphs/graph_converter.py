from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import polars as pl
import numpy as np
from spektral.data import Graph

from unravel.utils.objects.default_graph_converter import DefaultGraphConverter
from .graph_settings import BasketballGraphSettings

from .features.node_features import compute_node_features
from .features.adjacency_matrix import compute_adjacency_matrix
from .features.edge_features import compute_edge_features


if False:
    # for type checking
    from unravel.basketball.dataset.dataset import BasketballDataset

@dataclass(repr=True)
class BasketballGraphConverter(DefaultGraphConverter):
    """
    Converter for BasketballDataset to Spektral Graphs using Polars expressions.
    """
    # Expression variables container
    _exprs_variables: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        # Initialize expression variables
        self._exprs_variables = {
            "node_feature_cols": ["x", "y", "vx", "vy", "speed", "acceleration"],
            "label_col": self.label_col,
            "graph_id_col": self.graph_id_col,
        }
        super().__post_init__()

    def __init__(
        self,
        dataset: "BasketballDataset",
        settings: BasketballGraphSettings,
        label_col: str = "label",
        graph_id_col: str = "graph_id",
        graph_feature_cols: List[str] = None,
        **kwargs: Any,
    ):
        # Ensure graph_id and label columns exist on raw dataset
        if graph_id_col not in dataset.data.columns:
            dataset.add_graph_ids(by=["game_id", "event_id", "frame_id"], column_name=graph_id_col)
        if label_col not in dataset.data.columns and not kwargs.get("prediction", False):
            dataset.add_dummy_labels(by=["game_id", "event_id", "frame_id"], column_name=label_col)
        # Initialize base converter with common arguments
        super().__init__(
            label_col=label_col,
            graph_id_col=graph_id_col,
            **kwargs,
        )
        # Attach the dataframe and dataset object
        self.dataset = dataset.data
        self.dataset_obj = dataset
        self.settings = settings
        # Initialize expression variables container
        self._exprs_variables = {
            "node_feature_cols": graph_feature_cols or ["x", "y", "vx", "vy", "speed", "acceleration"],
            "label_col": label_col,
            "graph_id_col": graph_id_col,
        }

    def compute(self) -> pl.DataFrame:
        """
        Main conversion entry: checks, settings, then convert.
        """
        self._sport_specific_checks()
        self._settings_map = self._apply_settings()
        return self._convert()

    def _apply_settings(self) -> Dict[str, Any]:
        s = self.settings
        return {
            "ball_carrier_threshold": s.ball_carrier_threshold,
            "self_loop_ball": s.self_loop_ball,
            "adjacency_matrix_type": s.adjacency_matrix_type,
            "adjacency_matrix_connect_type": s.adjacency_matrix_connect_type,
            "pad": s.pad,
            "random_seed": s.random_seed,
            "label_type": s.label_type,
        }

    def _sport_specific_checks(self) -> None:
        df = self.dataset_obj.data
        if df is None or not isinstance(df, pl.DataFrame):
            raise ValueError("Dataset data must be a Polars DataFrame.")
        if self._exprs_variables["label_col"] not in df.columns and not self.prediction:
            raise ValueError(f"Missing label column '{self._exprs_variables['label_col']}'")
        if self._exprs_variables["graph_id_col"] not in df.columns:
            raise ValueError(f"Missing graph_id column '{self._exprs_variables['graph_id_col']}'")

    def _convert(self) -> pl.DataFrame:
        """
        Convert the raw Polars DataFrame into a graph‐structured DataFrame,
        returning one row per unique frame_id with columns:
          - id: frame identifier
          - x: node feature matrix (np.ndarray)
          - a: adjacency matrix (np.ndarray)
          - e: edge feature matrix (np.ndarray)
          - y: label for that frame

        Uses Polars group_by/agg to collect per-frame lists without Python-level loops.
        """
        from .features.node_features import compute_node_features
        from .features.adjacency_matrix import compute_adjacency_matrix
        from .features.edge_features import compute_edge_features

        df = self.dataset_obj.data
        node_cols = self._exprs_variables["node_feature_cols"]
        label_col = self._exprs_variables["label_col"]

        # Group by frame, collect each feature and team into list columns, grab first label
        aggregated = (
            df
            .group_by("frame_id")
            .agg(
                # For each node feature, pl.col(c) automatically produces a list of values
                *[pl.col(c).alias(f"{c}_list") for c in node_cols],
                pl.col("team").alias("team_list"),
                pl.col(label_col).first().alias("y"),
            )
        )

        # Build out each graph row from the collected lists
        rows = []
        for row in aggregated.rows(named=True):
            # Reconstruct per-entity dicts from parallel lists
            n = len(row[f"{node_cols[0]}_list"])
            records = [
                {c: row[f"{c}_list"][i] for c in node_cols} | {"team": row["team_list"][i]}
                for i in range(n)
            ]

            # Compute node matrix & teams
            x, teams = compute_node_features(
                records,
                normalize_coordinates=self.settings.normalize_coordinates,
                pitch_dimensions=self.settings.pitch_dimensions,
                node_feature_cols=node_cols,
            )
            # Build adjacency & edge
            a = compute_adjacency_matrix(teams, self_loop=self.settings.self_loop_ball)
            e = compute_edge_features(x)

            rows.append({
                "id": row["frame_id"],
                "x": x,
                "a": a,
                "e": e,
                "y": row["y"],
            })

        return pl.DataFrame(rows)




    def to_graph_frames(self) -> list[dict]:
        """
        Compute all graphs once and return a list of lightweight dicts:
        {"id": frame_id, "x": np.ndarray, "a": np.ndarray, "e": np.ndarray, "y": label}
        """
        if not hasattr(self, "_graph_df"):
            self._graph_df = self.compute()        # cache result

        return [
            {
                "id": row["id"],
                "x": row["x"],
                "a": row["a"],
                "e": row["e"],
                "y": row["y"],
            }
            for row in self._graph_df.rows(named=True)
        ]


    def to_spektral_graphs(self) -> list[Graph]:
        """
        Convert each frame dict into a Spektral Graph object.
        """
        return [
            Graph(
                x=frame["x"],
                a=frame["a"],
                e=frame["e"],
                y=np.asarray([frame["y"]], dtype=float),
                id=frame["id"],
            )
            for frame in self.to_graph_frames()
        ]
