from .utils import (
    normalize_speeds,
    normalize_sincos,
    normalize_distance,
    normalize_between,
    unit_vectors,
    normalize_angles,
    angle_between,
    normalize_speed_differences,
    graph_feature,
)

import numpy as np


@graph_feature(is_custom=False, feature_type="node")
def x_normed(**kwargs):
    return normalize_between(
        value=kwargs["x"],
        max_value=kwargs["settings"].pitch_dimensions.x_dim.max,
        min_value=kwargs["settings"].pitch_dimensions.x_dim.min,
    )


@graph_feature(is_custom=False, feature_type="node")
def y_normed(**kwargs):
    return normalize_between(
        value=kwargs["y"],
        max_value=kwargs["settings"].pitch_dimensions.y_dim.max,
        min_value=kwargs["settings"].pitch_dimensions.y_dim.min,
    )


@graph_feature(is_custom=False, feature_type="node")
def speeds_normed(**kwargs):
    return normalize_speeds(
        v=kwargs["v"],
        team_id=kwargs["team_id"],
        ball_id=kwargs["ball_id"],
        settings=kwargs["settings"],
    )


@graph_feature(is_custom=False, feature_type="node")
def velocity_components_2d_normed(**kwargs):
    uv_velocity = unit_vectors(vectors=kwargs["velocity"])
    angles = normalize_angles(np.arctan2(uv_velocity[:, 1], uv_velocity[:, 0]))
    return np.column_stack(
        (normalize_sincos(np.sin(angles)), normalize_sincos(np.cos(angles)))
    )


@graph_feature(is_custom=False, feature_type="node")
def distance_to_goal_normed(**kwargs):
    dist_to_goal = np.linalg.norm(
        kwargs["position"] - kwargs["settings"].goal_mouth_position, axis=1
    )
    return normalize_distance(
        value=dist_to_goal, max_distance=kwargs["settings"].max_goal_distance
    )


@graph_feature(is_custom=False, feature_type="node")
def distance_to_ball_normed(**kwargs):
    value = np.linalg.norm(kwargs["position"] - kwargs["ball_position"], axis=1)
    return normalize_distance(value=value, max_distance=kwargs["settings"].max_distance)


@graph_feature(is_custom=False, feature_type="node")
def is_possession_team(**kwargs):
    return np.where(
        kwargs["team_id"] == kwargs["possession_team_id"],
        1,
        kwargs["settings"].defending_team_node_value,
    )


@graph_feature(is_custom=False, feature_type="node")
def is_gk(**kwargs):
    return np.where(kwargs["is_gk"], 1, 0.1)


@graph_feature(is_custom=False, feature_type="node")
def is_ball(**kwargs):
    return np.where(kwargs["team_id"] == kwargs["ball_id"], 1, 0.1)


@graph_feature(is_custom=False, feature_type="node")
def angle_to_goal_components_2d_normed(**kwargs):
    vec_to_goal = kwargs["settings"].goal_mouth_position - kwargs["position"]
    angle_to_goal = np.arctan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    return np.column_stack(
        (
            normalize_sincos(np.sin(angle_to_goal)),
            normalize_sincos(np.cos(angle_to_goal)),
        )
    )


@graph_feature(is_custom=False, feature_type="node")
def angle_to_ball_components_2d_normed(**kwargs):
    vec_to_ball = kwargs["ball_position"] - kwargs["position"]
    angle_to_ball = np.arctan2(vec_to_ball[:, 1], vec_to_ball[:, 0])
    return np.column_stack(
        (
            normalize_sincos(np.sin(angle_to_ball)),
            normalize_sincos(np.cos(angle_to_ball)),
        )
    )


@graph_feature(is_custom=False, feature_type="node")
def angle_to_ball_normed(**kwargs):
    vec_to_ball = kwargs["ball_position"] - kwargs["position"]
    angle_to_ball = np.arctan2(vec_to_ball[:, 1], vec_to_ball[:, 0])
    return np.column_stack(
        (
            normalize_sincos(np.sin(angle_to_ball)),
            normalize_sincos(np.cos(angle_to_ball)),
        )
    )


@graph_feature(is_custom=False, feature_type="node")
def is_ball_carrier(**kwargs):
    return np.where((kwargs["is_ball_carrier"]), 1, 0.1)


@graph_feature(is_custom=False, feature_type="edge")
def distances_between_players_normed(**kwargs):
    distances_between_players = np.linalg.norm(
        kwargs["position"][:, None, :] - kwargs["position"][None, :, :], axis=-1
    )
    return normalize_distance(
        distances_between_players, max_distance=kwargs["settings"].max_distance
    )


@graph_feature(is_custom=False, feature_type="edge")
def speed_difference_normed(**kwargs):
    speed_diff_matrix = np.nan_to_num(kwargs["v"][None, :] - kwargs["v"][:, None])
    # overwrite 'v' temporarily from (N,) array to (N, N) array
    kwargs["v"] = speed_diff_matrix
    return normalize_speed_differences(**kwargs)


@graph_feature(is_custom=False, feature_type="edge")
def angle_between_players_normed(**kwargs):
    position_2d = kwargs["position"][:, :2]
    vect_to_player_matrix = position_2d[:, None, :] - position_2d[None, :, :]

    angle_pos_matrix = np.nan_to_num(
        np.arctan2(vect_to_player_matrix[:, :, 1], vect_to_player_matrix[:, :, 0])
    )
    return (
        normalize_sincos(np.nan_to_num(np.cos(angle_pos_matrix))),
        normalize_sincos(np.nan_to_num(np.sin(angle_pos_matrix))),
    )


@graph_feature(is_custom=False, feature_type="edge")
def velocity_difference_normed(**kwargs):
    position_2d = kwargs["position"][:, :2]
    vect_to_player_matrix = position_2d[:, None, :] - position_2d[None, :, :]

    v_normed_matrix = kwargs["velocity"][None, :, :] - kwargs["velocity"][:, None, :]

    combined_matrix = np.concatenate((vect_to_player_matrix, v_normed_matrix), axis=2)
    angle_vel_matrix = np.apply_along_axis(angle_between, 2, combined_matrix)
    return (
        normalize_sincos(np.nan_to_num(np.cos(angle_vel_matrix))),
        normalize_sincos(np.nan_to_num(np.sin(angle_vel_matrix))),
    )
