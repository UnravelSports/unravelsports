import numpy as np

from .utils import normalize_speed, normalize_distance, normalize_sincos


class EdgeFeatureSet:
    """
    To manage and store edge feature functions configured by user
    """

    def __init__(self):
        self.edge_feature_functions = []

    def add_dist_matrix(self, normed: bool = True):
        if normed:
            self.edge_feature_functions.append(
                (
                    "normalize_dist",
                    normalize_distance,
                    ["distances_between_players", "max_dist_to_player"],
                )
            )
        else:
            self.edge_feature_functions.append(
                ("raw_dist", lambda dist: dist, ["distances_between_players"])
            )

        return self

    def add_speed_diff_matrix(self, normed: bool = True):
        if normed:
            self.edge_feature_functions.append(
                (
                    "normalize_speed_diff",
                    lambda a_spe, h_spe, max_speed: np.nan_to_num(
                        normalize_speed(a_spe[None, :], max_speed)
                        - normalize_speed(h_spe[:, None], max_speed)
                    ),
                    ["a_spe", "h_spe", "max_speed"],
                )
            )
        else:
            self.edge_feature_functions.append(
                (
                    "raw_speed_diff",
                    lambda a_spe, h_spe: np.nan_to_num(a_spe[None, :] - h_spe[:, None]),
                    ["a_spe", "h_spe"],
                )
            )

        return self

    def add_pos_cos_matrix(self, normed: bool = True):
        if normed:
            self.edge_feature_functions.append(
                (
                    "normalise_cos_pos",
                    lambda angle_pos_matrix: normalize_sincos(
                        np.nan_to_num(np.cos(angle_pos_matrix))
                    ),
                    ["angle_pos_matrix"],
                )
            )
        else:
            self.edge_feature_functions.append(
                (
                    "raw_cos_pos",
                    lambda angle_pos_matrix: np.nan_to_num(np.cos(angle_pos_matrix)),
                    ["angle_pos_matrix"],
                )
            )

        return self

    def add_pos_sin_matrix(self, normed: bool = True):
        if normed:
            self.edge_feature_functions.append(
                (
                    "normalise_sin_pos",
                    lambda angle_pos_matrix: normalize_sincos(
                        np.nan_to_num(np.sin(angle_pos_matrix))
                    ),
                    ["angle_pos_matrix"],
                )
            )
        else:
            self.edge_feature_functions.append(
                (
                    "raw_sin_pos",
                    lambda angle_pos_matrix: np.nan_to_num(np.sin(angle_pos_matrix)),
                    ["angle_pos_matrix"],
                )
            )

        return self

    def add_vel_cos_matrix(self, normed: bool = True):
        if normed:
            self.edge_feature_functions.append(
                (
                    "normalise_cos_vel",
                    lambda angle_vel_matrix: normalize_sincos(
                        np.nan_to_num(np.cos(angle_vel_matrix))
                    ),
                    ["angle_vel_matrix"],
                )
            )
        else:
            self.edge_feature_functions.append(
                (
                    "raw_cos_vel",
                    lambda angle_vel_matrix: np.nan_to_num(np.cos(angle_vel_matrix)),
                    ["angle_vel_matrix"],
                )
            )

        return self

    def add_vel_sin_matrix(self, normed: bool = True):
        if normed:
            self.edge_feature_functions.append(
                (
                    "normalise_sin_vel",
                    lambda angle_vel_matrix: normalize_sincos(
                        np.nan_to_num(np.sin(angle_vel_matrix))
                    ),
                    ["angle_vel_matrix"],
                )
            )
        else:
            self.edge_feature_functions.append(
                (
                    "raw_sin_vel",
                    lambda angle_vel_matrix: np.nan_to_num(np.sin(angle_vel_matrix)),
                    ["angle_vel_matrix"],
                )
            )

        return self

    def get_features(self):
        return self.edge_feature_functions
