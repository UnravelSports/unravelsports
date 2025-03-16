import math
import numpy as np

from ....utils import (
    normalize_coords,
    normalize_speeds_nfl,
    normalize_sincos,
    normalize_distance,
    unit_vector_from_angle,
    normalize_speeds_nfl,
    normalize_accelerations_nfl,
    normalize_between,
    unit_vector,
    unit_vectors,
    normalize_angles,
    normalize_distance,
    normalize_coords,
    normalize_speed,
    distance_to_ball,
)
from ...dataset.kloppy_polars import Constant
from .node_feature_func_map import get_node_feature_func_map


def compute_node_features_pl(
    x,
    y,
    s,
    velocity,
    team,
    possession_team,
    is_gk,
    ball_carrier,
    graph_features,
    settings,
    feature_dict,
):
    ball_id = Constant.BALL

    if len(np.where(team == ball_id)) >= 1:
        ball_index = np.where(team == ball_id)[0]
    else:
        ball_index = 0

    feature_func_map = get_node_feature_func_map(
        x,
        y,
        s,
        velocity,
        team,
        possession_team,
        is_gk,
        ball_carrier,
        graph_features,
        settings,
    )
    computed_features = []

    for feature, custom_params in feature_dict.items():
        if feature in feature_func_map:
            params = feature_func_map[feature]["defaults"].copy()
            params.update(custom_params)
            computed_features.append(feature_func_map[feature]["func"](**params))
        else:
            print("Error in feature", feature)
    X = np.nan_to_num(np.stack(computed_features, axis=-1))

    if graph_features is not None:
        eg = np.ones((X.shape[0], graph_features.shape[0])) * 0.0
        eg[ball_index] = graph_features
        X = np.hstack((X, eg))

    return X
