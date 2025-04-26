import numpy as np
from typing import List, Tuple, Any
from unravel.basketball.graphs.graph_settings import BasketballPitchDimensions


def compute_node_features(
    records: List[dict],
    normalize_coordinates: bool,
    pitch_dimensions: BasketballPitchDimensions,
    node_feature_cols: List[str] = None,
) -> Tuple[np.ndarray, List[Any]]:
    """
    Build the node feature matrix and extract team labels.

    Args:
        records: List of dicts, each representing one entity in the frame, e.g.:
            {
              "x": float,
              "y": float,
              "vx": float,
              "vy": float,
              "speed": float,
              "acceleration": float,
              "team": Any,
              ...
            }
        normalize_coordinates: If True, scale x by court_length and y by court_width.
        pitch_dimensions: BasketballPitchDimensions instance containing court dimensions.
        node_feature_cols: List of keys from each record to include as features, in order.
            Defaults to ["x", "y", "vx", "vy", "speed", "acceleration"].

    Returns:
        x_array: NumPy array of shape (n_nodes, n_node_features) with node features.
        teams: List of length n_nodes containing the team label for each node.
    """
    # Use default feature list if none provided
    if node_feature_cols is None:
        node_feature_cols = ["x", "y", "vx", "vy", "speed", "acceleration"]

    x_list: List[List[float]] = []
    teams: List[Any] = []

    for rec in records:
        features: List[float] = []
        for col in node_feature_cols:
            # Retrieve raw value (might be None)
            val = rec.get(col, 0.0)
            # Coerce None → 0.0 to avoid float(None)
            if val is None:
                val = 0.0

            # If normalizing and the feature is a coordinate, scale it
            if normalize_coordinates and col in ("x", "y"):
                if col == "x":
                    val = val / pitch_dimensions.court_length
                else:  # col == "y"
                    val = val / pitch_dimensions.court_width

            features.append(float(val))

        x_list.append(features)
        # Collect the team label for adjacency construction
        teams.append(rec.get("team"))

    # Convert list of feature lists to a 2D NumPy array
    x_array = np.asarray(x_list, dtype=float)
    return x_array, teams
