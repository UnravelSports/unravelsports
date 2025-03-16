import numpy as np

from ....utils import (
    normalize_distance,
    normalize_speed,
    normalize_sincos,
    angle_between,
    non_zeros,
    reindex,
)

import numpy as np

from ....utils import (
    normalize_distance,
    normalize_sincos,
    non_zeros,
    reindex,
    normalize_speed_differences_nfl,
    normalize_accelerations_nfl,
)

from ...dataset.kloppy_polars import Constant


def compute_edge_features_pl(
    adjacency_matrix, p3d, p2d, s, velocity, team, settings, feature_dict
):
    # Compute pairwise distances using broadcasting
    max_dist_to_player = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    )

    distances_between_players = np.linalg.norm(
        p3d[:, None, :] - p3d[None, :, :], axis=-1
    )
    # dist_matrix_normed = normalize_distance(
    #     distances_between_players, max_distance=max_dist_to_player
    # )  # 11x11

    # speed_diff_matrix = np.nan_to_num(s[None, :] - s[:, None])  # NxNx1
    # speed_diff_matrix_normed = normalize_speed_differences_nfl(
    #     s=speed_diff_matrix,
    #     team=team,
    #     ball_id=Constant.BALL,
    #     settings=settings,
    # )

    # vect_to_player_matrix = p2d[:, None, :] - p2d[None, :, :]  # NxNx2

    v_normed_matrix = velocity[None, :, :] - velocity[:, None, :]  # 11x11x2

    vect_to_player_matrix = (
        p2d[:, None, :] - p2d[None, :, :]
    )  # 11x11x2 the vector between two players

    # Angles between players in sin and cos
    # angle_pos_matrix = np.nan_to_num(
    #     np.arctan2(vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0])
    # )
    # pos_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix)))
    # pos_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix)))

    combined_matrix = np.concatenate((vect_to_player_matrix, v_normed_matrix), axis=2)
    angle_vel_matrix = np.apply_along_axis(angle_between, 2, combined_matrix)
    # vel_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_vel_matrix)))
    # vel_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_vel_matrix)))

    nan_mask = np.isnan(distances_between_players)
    non_zero_idxs, len_a = non_zeros(A=adjacency_matrix)

    # dist_matrix_normed[nan_mask] = 0
    # speed_diff_matrix_normed[nan_mask] = 0

    # pos_cos_matrix[nan_mask] = 0
    # pos_sin_matrix[nan_mask] = 0
    feature_func_map = {
        # """
        #     distances_between_players = np.linalg.norm(
        #         p3d[:, None, :] - p3d[None, :, :], axis=-1
        #     )
        #     dist_matrix_normed = normalize_distance(
        #         distances_between_players, max_distance=max_dist_to_player
        #         )  # 11x11
        #     dist_matrix_normed[nan_mask] = 0
        # """
        "dist_matrix": {
            "func": lambda value: value,
            "defaults": {
                "value": np.linalg.norm(p3d[:, None, :] - p3d[None, :, :], axis=-1)
            },
        },
        "dist_matrix_normed": {
            "func": lambda value, max_distance: np.where(
                nan_mask, 0, normalize_distance(value, max_distance)
            ),
            "defaults": {
                "value": np.linalg.norm(p3d[:, None, :] - p3d[None, :, :], axis=-1),
                "max_distance": max_dist_to_player,
            },
        },
        # """
        #     speed_diff_matrix = np.nan_to_num(s[None, :] - s[:, None])  # NxNx1
        #     speed_diff_matrix_normed = normalize_speed_differences_nfl(
        #         s=speed_diff_matrix,
        #         team=team,
        #         ball_id=Constant.BALL,
        #         settings=settings,
        #     )
        #     speed_diff_matrix_normed[nan_mask] = 0
        # """
        "speed_diff_matrix": {
            "func": lambda value: value,
            "defaults": {"value": np.nan_to_num(s[None, :] - s[:, None])},
        },
        "speed_diff_matrix_normed": {
            "func": lambda s, team, ball_id, settings: np.where(
                nan_mask, 0, normalize_speed_differences_nfl(s, team, ball_id, settings)
            ),
            "defaults": {
                "s": np.nan_to_num(s[None, :] - s[:, None]),
                "team": team,
                "ball_id": Constant.BALL,
                "settings": settings,
            },
        },
        # """
        #     vect_to_player_matrix = (
        #         p2d[:, None, :] - p2d[None, :, :]
        #     )  # 11x11x2 the vector between two players
        #     # Angles between players in sin and cos
        #     angle_pos_matrix = np.nan_to_num(
        #         np.arctan2(vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0])
        #     )
        #     pos_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix)))
        #     pos_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix)))
        #     pos_cos_matrix[nan_mask] = 0
        #     pos_sin_matrix[nan_mask] = 0
        # """
        "angle_pos_matrix": {
            "func": lambda vect_to_player_matrix: np.where(
                nan_mask,
                0,
                np.arctan2(
                    vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0]
                ),
            ),
            "defaults": {"vect_to_player_matrix": vect_to_player_matrix},
        },
        "pos_cos_matrix": {
            "func": lambda angle_pos_matrix: np.where(
                nan_mask, 0, normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix)))
            ),
            "defaults": {
                "angle_pos_matrix": np.nan_to_num(
                    np.arctan2(
                        vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0]
                    )
                )
            },
        },
        "pos_sin_matrix": {
            "func": lambda angle_pos_matrix: np.where(
                nan_mask, 0, normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix)))
            ),
            "defaults": {
                "angle_pos_matrix": np.nan_to_num(
                    np.arctan2(
                        vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0]
                    )
                )
            },
        },
        # """
        #     v_normed_matrix = velocity[None, :, :] - velocity[:, None, :]  # 11x11x2
        #     vect_to_player_matrix = (
        #             p2d[:, None, :] - p2d[None, :, :]
        #         )  # 11x11x2 the vector between two players
        #     combined_matrix = np.concatenate((vect_to_player_matrix, v_normed_matrix), axis=2)
        #     angle_vel_matrix = np.apply_along_axis(angle_between, 2, combined_matrix)
        #     vel_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_vel_matrix)))
        #     vel_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_vel_matrix)))
        # """
        "angle_vel_matrix": {
            "func": lambda angle_between, combined_matrix: np.where(
                nan_mask, 0, np.apply_along_axis(angle_between, 2, combined_matrix)
            ),
            "defaults": {
                "angle_between": angle_between,
                "combined_matrix": combined_matrix,
            },
        },
        "vel_cos_matrix": {
            "func": lambda angle_vel_matrix: normalize_sincos(
                np.nan_to_num(np.cos(angle_vel_matrix))
            ),
            "defaults": {"angle_vel_matrix": angle_vel_matrix},
        },
        "vel_sin_matrix": {
            "func": lambda angle_vel_matrix: normalize_sincos(
                np.nan_to_num(np.sin(angle_vel_matrix))
            ),
            "defaults": {"angle_vel_matrix": angle_vel_matrix},
        },
    }

    computed_features = []

    for feature, custom_params in feature_dict.items():
        if feature in feature_func_map:
            params = feature_func_map[feature]["defaults"].copy()
            params.update(custom_params)
            computed_value = feature_func_map[feature]["func"](**params)
            computed_features.append(reindex(computed_value, non_zero_idxs, len_a))

    e_tuple = list(computed_features)
    e = np.concatenate(e_tuple, axis=1)

    return np.nan_to_num(e)
