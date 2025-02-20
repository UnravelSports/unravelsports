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


def compute_edge_features_pl(adjacency_matrix, p3d, p2d, s, velocity, team, settings, feature_dict):
    # Compute pairwise distances using broadcasting
    max_dist_to_player = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    )

    distances_between_players = np.linalg.norm(
        p3d[:, None, :] - p3d[None, :, :], axis=-1
    )
    dist_matrix_normed = normalize_distance(
        distances_between_players, max_distance=max_dist_to_player
    )  # 11x11

    speed_diff_matrix = np.nan_to_num(s[None, :] - s[:, None])  # NxNx1
    speed_diff_matrix_normed = normalize_speed_differences_nfl(
        s=speed_diff_matrix,
        team=team,
        ball_id=Constant.BALL,
        settings=settings,
    )

    vect_to_player_matrix = p2d[:, None, :] - p2d[None, :, :]  # NxNx2

    v_normed_matrix = velocity[None, :, :] - velocity[:, None, :]  # 11x11x2

    vect_to_player_matrix = (
        p2d[:, None, :] - p2d[None, :, :]
    )  # 11x11x2 the vector between two players

    # Angles between players in sin and cos
    angle_pos_matrix = np.nan_to_num(
        np.arctan2(vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0])
    )
    pos_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix)))
    pos_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix)))

    combined_matrix = np.concatenate((vect_to_player_matrix, v_normed_matrix), axis=2)
    angle_vel_matrix = np.apply_along_axis(angle_between, 2, combined_matrix)
    vel_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_vel_matrix)))
    vel_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_vel_matrix)))

    nan_mask = np.isnan(distances_between_players)
    non_zero_idxs, len_a = non_zeros(A=adjacency_matrix)

    dist_matrix_normed[nan_mask] = 0
    speed_diff_matrix_normed[nan_mask] = 0

    pos_cos_matrix[nan_mask] = 0
    pos_sin_matrix[nan_mask] = 0

    e_tuple = list(
        [
            reindex(dist_matrix_normed, non_zero_idxs, len_a),
            reindex(speed_diff_matrix_normed, non_zero_idxs, len_a),
            reindex(pos_cos_matrix, non_zero_idxs, len_a),
            reindex(pos_sin_matrix, non_zero_idxs, len_a),
            reindex(vel_cos_matrix, non_zero_idxs, len_a),
            reindex(vel_sin_matrix, non_zero_idxs, len_a),
        ]
    )

    e = np.concatenate(e_tuple, axis=1)
    return np.nan_to_num(e)


# def edge_features(
#     attacking_players,
#     defending_players,
#     ball,
#     max_player_speed,
#     max_ball_speed,
#     pitch_dimensions,
#     adjacency_matrix,
#     delaunay_adjacency_matrix,
# ):
#     """
#     # edge features matrix is (np.non_zero(a), n_edge_features) (nz, n_edge_features)
#     # so for every connected edge in the adjacency matrix (a) we have 1 row of features describing that edge
#     # to do this we compute all values for a single feature in a <=23x23 square matrix
#     # reshape it to a (<=23**2, ) matrix and then mask all values that are 0 in `a` (nz)
#     # then we concat all the features into a single (nz, n_edge_features) matrix
#     """

#     max_dist_to_player = np.sqrt(
#         pitch_dimensions.pitch_length**2 + pitch_dimensions.pitch_width**2
#     )

#     players1 = players2 = attacking_players + defending_players + [ball]

#     h_pos = np.asarray([p.position for p in players1])
#     a_pos = np.asarray([p.position for p in players2])

#     h_vel = np.asarray([p.velocity for p in players1])
#     a_vel = np.asarray([p.velocity for p in players2])

#     h_spe = np.asarray([p.speed for p in players1])
#     a_spe = np.asarray([p.speed for p in players2])

#     distances_between_players = np.linalg.norm(
#         h_pos[:, None, :] - a_pos[None, :, :], axis=-1
#     )
#     nan_mask = np.isnan(distances_between_players)

#     dist_matrix = normalize_distance(
#         distances_between_players, max_distance=max_dist_to_player
#     )  # 11x11

#     speed_diff_matrix = np.nan_to_num(
#         normalize_speed(a_spe[None, :], max_speed=max(max_player_speed, max_ball_speed))
#         - normalize_speed(
#             h_spe[:, None], max_speed=max(max_player_speed, max_ball_speed)
#         )
#     )  # 11x11x1

#     vect_to_player_matrix = (
#         h_pos[:, None, :] - a_pos[None, :, :]
#     )  # 11x11x2 the vector between two players
#     v_normed_matrix = a_vel[None, :, :] - h_vel[:, None, :]  # 11x11x2

#     angle_pos_matrix = np.nan_to_num(
#         np.arctan2(vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0])
#     )
#     pos_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix)))
#     pos_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix)))

#     combined_matrix = np.concatenate((vect_to_player_matrix, v_normed_matrix), axis=2)
#     angle_vel_matrix = np.apply_along_axis(angle_between, 2, combined_matrix)
#     vel_cos_matrix = normalize_sincos(np.nan_to_num(np.cos(angle_vel_matrix)))
#     vel_sin_matrix = normalize_sincos(np.nan_to_num(np.sin(angle_vel_matrix)))

#     non_zero_idxs, len_a = non_zeros(A=adjacency_matrix)
#     # create a matrix where 1 if edge is same team else 0

#     # if we have nan values we mask them to 0.
#     # this only happens when we pad additional players
#     dist_matrix[nan_mask] = 0
#     speed_diff_matrix[nan_mask] = 0
#     pos_cos_matrix[nan_mask] = 0
#     pos_sin_matrix[nan_mask] = 0
#     vel_cos_matrix[nan_mask] = 0
#     vel_sin_matrix[nan_mask] = 0

#     e_tuple = list(
#         [
#             # same_team_matrix[non_zero_idxs].reshape(len_a, 1),
#             reindex(dist_matrix, non_zero_idxs, len_a),
#             reindex(speed_diff_matrix, non_zero_idxs, len_a),
#             reindex(pos_cos_matrix, non_zero_idxs, len_a),
#             reindex(pos_sin_matrix, non_zero_idxs, len_a),
#             reindex(vel_cos_matrix, non_zero_idxs, len_a),
#             reindex(vel_sin_matrix, non_zero_idxs, len_a),
#         ]
#     )

#     if delaunay_adjacency_matrix is not None:
#         # if we are not using Delaunay as adjacency matrix,
#         # use it as edge features to indicate "clear passing lines"
#         extra_tuple = list([reindex(delaunay_adjacency_matrix, non_zero_idxs, len_a)])
#         e_tuple.extend(extra_tuple)

#     e = np.concatenate(e_tuple, axis=1)
#     return np.nan_to_num(e)
