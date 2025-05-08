import numpy as np


from ....utils import (
    normalize_coords,
    normalize_speeds,
    normalize_sincos,
    normalize_distance,
    unit_vector_from_angle,
    normalize_speeds,
    normalize_accelerations_nfl,
    normalize_between,
)

from ...dataset import Constant


def compute_node_features(
    x,
    y,
    s,
    a,
    o,
    dir,
    team,
    official_position,
    possession_team,
    height,
    weight,
    graph_features,
    settings,
):
    ball_id = Constant.BALL

    goal_mouth_position = (
        settings.pitch_dimensions.x_dim.max,
        (settings.pitch_dimensions.y_dim.max + settings.pitch_dimensions.y_dim.min) / 2,
    )
    max_dist_to_player = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    )
    max_dist_to_goal = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    )

    position = np.stack((x, y), axis=-1)

    if len(np.where(team == ball_id)) >= 1:
        ball_index = np.where(team == ball_id)[0]
        ball_position = position[ball_index][0]
    else:
        ball_position = np.asarray([np.nan, np.nan])
        ball_index = 0

    x_normed = normalize_between(
        value=x,
        max_value=settings.pitch_dimensions.x_dim.max,
        min_value=settings.pitch_dimensions.x_dim.min,
    )
    y_normed = normalize_between(
        value=y,
        max_value=settings.pitch_dimensions.y_dim.max,
        min_value=settings.pitch_dimensions.y_dim.min,
    )
    uv_sa = unit_vector_from_angle(value=s, angle_radians=dir)
    s_normed = normalize_speeds(s, team, ball_id=Constant.BALL, settings=settings)

    uv_aa = unit_vector_from_angle(value=a, angle_radians=dir)
    a_normed = normalize_accelerations_nfl(
        a, team, ball_id=Constant.BALL, settings=settings
    )

    dir_sin_normed = normalize_sincos(np.nan_to_num(np.sin(dir)))
    dir_cos_normed = normalize_sincos(np.nan_to_num(np.cos(dir)))
    o_sin_normed = normalize_sincos(np.nan_to_num(np.sin(o)))
    o_cos_normed = normalize_sincos(np.nan_to_num(np.cos(o)))

    dist_to_goal = np.linalg.norm(position - goal_mouth_position, axis=1)
    normed_dist_to_goal = normalize_distance(
        value=dist_to_goal, max_distance=max_dist_to_goal
    )

    dist_to_ball = np.linalg.norm(position - ball_position, axis=1)
    normed_dist_to_ball = normalize_distance(
        value=dist_to_ball, max_distance=max_dist_to_player
    )

    dist_to_end_zone = settings.pitch_dimensions.end_zone - x
    normed_dist_to_end_zone = normalize_between(
        value=dist_to_end_zone,
        max_value=settings.pitch_dimensions.pitch_length,
        min_value=0,
    )

    is_possession_team = np.where(
        team == possession_team, 1, settings.defending_team_node_value
    )
    is_qb = np.where(
        official_position == Constant.QB,  # First condition
        1,  # If true, set to 1 (indicating the player is a QB)
        np.where(
            team == possession_team,  # Second condition inside the else of the first
            settings.attacking_non_qb_node_value,  # If true, set to attacking_non_qb_value
            0,  # If false, set to 0
        ),
    )

    is_ball = np.where(team == ball_id, 1, 0)
    weight_normed = normalize_between(
        min_value=settings.min_weight, max_value=settings.max_weight, value=weight
    )
    height_normed = normalize_between(
        min_value=settings.min_height, max_value=settings.max_height, value=height
    )

    X = np.nan_to_num(
        np.stack(
            (
                x_normed,
                y_normed,
                uv_sa[0],
                uv_sa[1],
                s_normed,
                uv_aa[0],
                uv_aa[1],
                a_normed,
                dir_sin_normed,
                dir_cos_normed,
                o_sin_normed,
                o_cos_normed,
                normed_dist_to_goal,
                normed_dist_to_ball,
                normed_dist_to_end_zone,
                is_possession_team,
                is_qb,
                is_ball,
                weight_normed,
                height_normed,
            ),
            axis=-1,
        )
    )

    if graph_features is not None:
        eg = np.ones((X.shape[0], graph_features.shape[0])) * 0.0
        eg[ball_index] = graph_features
        X = np.hstack((X, eg))

    return X
