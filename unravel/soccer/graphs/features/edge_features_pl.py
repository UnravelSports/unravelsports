import numpy as np

from ....utils import (
    normalize_distance,
    normalize_speed,
    normalize_sincos,
    angle_between,
    non_zeros,
    reindex,
)

from ...dataset.kloppy_polars import Constant

from .edge_feature_func_map import get_edge_feature_func_map


def compute_edge_features_pl(
    adjacency_matrix, p3d, p2d, s, velocity, team, settings, feature_dict
):

    non_zero_idxs, len_a = non_zeros(A=adjacency_matrix)

    feature_func_map = get_edge_feature_func_map(p3d, p2d, s, velocity, team, settings)
    computed_features = []

    for feature, custom_params in feature_dict.items():
        if feature in feature_func_map:
            params = feature_func_map[feature]["defaults"].copy()
            custom_params = {k: v for k, v in custom_params.items() if v is not None}
            params.update(custom_params)
            computed_value = feature_func_map[feature]["func"](**params)
            computed_features.append(reindex(computed_value, non_zero_idxs, len_a))

    e_tuple = list(computed_features)
    e = np.concatenate(e_tuple, axis=1)

    return np.nan_to_num(e)
