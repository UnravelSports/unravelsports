from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import polars as pl
import numpy as np
from spektral.data import Graph

from unravel.utils.objects.default_graph_converter import DefaultGraphConverter
from .graph_settings import BasketballGraphSettings

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
        df = self.dataset_obj.data
        # get all unique frame IDs
        frame_ids = df.select("frame_id").unique().to_series().to_list()

        rows = []
        for fid in frame_ids:
            recs = df.filter(pl.col("frame_id") == fid).to_dicts()
            x, teams = self._compute_node_features(recs)
            a = self._compute_adjacency(teams)
            e = self._compute_edge_features(x)
            y = recs[0].get(self.label_col)
            rows.append({"id": fid, "x": x, "a": a, "e": e, "y": y})

        return pl.DataFrame(rows)



    def _compute_node_features(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Any]]:
        xs, teams = [], []
        for rec in records:
            x, y = rec.get("x", 0.0), rec.get("y", 0.0)
            if self.settings.normalize_coordinates:
                x /= self.settings.pitch_dimensions.court_length
                y /= self.settings.pitch_dimensions.court_width
            xs.append([x, y])
            teams.append(rec.get("team"))
        return np.array(xs), teams

    def _compute_adjacency(self, teams: List[Any]) -> np.ndarray:
        arr = np.array(teams)
        A = (arr[:, None] == arr[None, :]).astype(float)
        if not self.settings.self_loop_ball:
            np.fill_diagonal(A, 0.0)
        return A

    def _compute_edge_features(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - x[None, :, :]
        return np.linalg.norm(diff, axis=2)
