import numpy as np

from ....utils import (
    normalize_distance,
    normalize_sincos,
    non_zeros,
    reindex,
    normalize_speed_differences_nfl,
    normalize_accelerations_nfl,
)


def compute_edge_features(adjacency_matrix, p, s, a, o, dir, team, settings):
    # Compute pairwise distances using broadcasting
    max_dist_to_player = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    )

    distances_between_players = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    dist_matrix_normed = normalize_distance(
        distances_between_players, max_distance=max_dist_to_player
    )  # 11x11

    speed_diff_matrix = np.nan_to_num(s[None, :] - s[:, None])  # NxNx1
    speed_diff_matrix_normed = normalize_speed_differences_nfl(
        s=speed_diff_matrix,
        team=team,
        settings=settings,
    )
    acc_diff_matrix = np.nan_to_num(a[None, :] - a[:, None])  # NxNx1
    acc_diff_matrix_normed = normalize_accelerations_nfl(
        a=acc_diff_matrix,
        team=team,
        settings=settings,
    )
    vect_to_player_matrix = p[:, None, :] - p[None, :, :]  # NxNx2

    # Angles between players in sin and cos
    angle_pos_matrix = np.nan_to_num(
        np.arctan2(vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0])
    )
    pos_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix)))
    pos_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix)))

    dir_diff_matrix = dir[None, :] - dir[:, None]  # NxNx1
    dir_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(dir_diff_matrix)))
    dir_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(dir_diff_matrix)))
    o_diff_matrix = o[None, :] - o[:, None]  # NxNx1
    o_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(o_diff_matrix)))
    o_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(o_diff_matrix)))

    nan_mask = np.isnan(distances_between_players)
    non_zero_idxs, len_a = non_zeros(A=adjacency_matrix)

    dist_matrix_normed[nan_mask] = 0
    speed_diff_matrix_normed[nan_mask] = 0
    acc_diff_matrix[nan_mask] = 0
    pos_cos_matrix[nan_mask] = 0
    pos_sin_matrix[nan_mask] = 0
    dir_cos_matrix[nan_mask] = 0
    dir_sin_matrix[nan_mask] = 0
    o_cos_matrix[nan_mask] = 0
    o_sin_matrix[nan_mask] = 0

    e_tuple = list(
        [
            reindex(dist_matrix_normed, non_zero_idxs, len_a),
            reindex(speed_diff_matrix_normed, non_zero_idxs, len_a),
            reindex(acc_diff_matrix_normed, non_zero_idxs, len_a),
            reindex(pos_cos_matrix, non_zero_idxs, len_a),
            reindex(pos_sin_matrix, non_zero_idxs, len_a),
            reindex(dir_cos_matrix, non_zero_idxs, len_a),
            reindex(dir_sin_matrix, non_zero_idxs, len_a),
            reindex(o_cos_matrix, non_zero_idxs, len_a),
            reindex(o_sin_matrix, non_zero_idxs, len_a),
        ]
    )

    e = np.concatenate(e_tuple, axis=1)
    return np.nan_to_num(e)
