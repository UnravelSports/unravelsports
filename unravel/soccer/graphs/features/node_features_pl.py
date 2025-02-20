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
    feature_dict
):
    ball_id = Constant.BALL

    position = np.stack((x, y), axis=-1)

    if len(np.where(team == ball_id)) >= 1:
        ball_index = np.where(team == ball_id)[0]
        ball_position = position[ball_index][0]
    else:
        ball_position = np.asarray([np.nan, np.nan])
        ball_index = 0

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

    position, ball_position, dist_to_ball = distance_to_ball(
        x=x, y=y, team=team, ball_id=ball_id
    )

    # x_normed = normalize_between(
    #     value=x,
    #     max_value=settings.pitch_dimensions.x_dim.max,
    #     min_value=settings.pitch_dimensions.x_dim.min,
    # )
    # y_normed = normalize_between(
    #     value=y,
    #     max_value=settings.pitch_dimensions.y_dim.max,
    #     min_value=settings.pitch_dimensions.y_dim.min,
    # )
    # s_normed = normalize_speeds_nfl(s, team, ball_id=ball_id, settings=settings)
    # uv_velocity = unit_vectors(velocity)

    # angles = normalize_angles(np.arctan2(uv_velocity[:, 1], uv_velocity[:, 0]))
    # v_sin_normed = normalize_sincos(np.sin(angles))
    # v_cos_normed = normalize_sincos(np.cos(angles))

    # dist_to_goal = np.linalg.norm(position - goal_mouth_position, axis=1)
    # normed_dist_to_goal = normalize_distance(
    #     value=dist_to_goal, max_distance=max_dist_to_goal
    # )

    # normed_dist_to_ball = normalize_distance(
    #     value=dist_to_ball, max_distance=max_dist_to_player
    # )

    # vec_to_goal = goal_mouth_position - position
    # angle_to_goal = np.arctan2(vec_to_goal[:, 1], vec_to_goal[:, 0])
    # goal_sin_normed = normalize_sincos(np.sin(angle_to_goal))
    # goal_cos_normed = normalize_sincos(np.cos(angle_to_goal))

    # vec_to_ball = ball_position - position
    # angle_to_ball = np.arctan2(vec_to_ball[:, 1], vec_to_ball[:, 0])
    # ball_sin_normed = normalize_sincos(np.sin(angle_to_ball))
    # ball_cos_normed = normalize_sincos(np.cos(angle_to_ball))

    # is_possession_team = np.where(
    #     team == possession_team, 1, settings.defending_team_node_value
    # )

    # is_ball = np.where(team == ball_id, 1, 0)
    feature_func_map = {
        'x': {'func': lambda value: value, 'defaults': {'value': x}},
        'x_normed': {'func': normalize_between, 'defaults': {'value': x, 'max_value': settings.pitch_dimensions.x_dim.max, 'min_value': settings.pitch_dimensions.x_dim.min}},
        'y': {'func': lambda value: value, 'defaults': {'value': y}},
        'y_normed': {'func': normalize_between, 'defaults': {'value': y, 'max_value': settings.pitch_dimensions.y_dim.max, 'min_value': settings.pitch_dimensions.y_dim.min}},
        's': {'func': lambda value: value, 'defaults': {'value': s}},
        's_normed': {'func': normalize_speeds_nfl, 'defaults': {'s': s, 'team': team, 'ball_id': ball_id, 'settings': settings}} 
    }
    computed_features = []
    for feature, custom_params in feature_dict.items():
        if feature in feature_func_map:
            params = feature_func_map[feature]['defaults'].copy()
            params.update(custom_params)
            computed_features.append(feature_func_map[feature]['func'](**params))
    X = np.nan_to_num(np.stack(computed_features, axis=-1))
    # X = np.nan_to_num(
    #     np.stack(
    #         (
    #             x_normed,
    #             y_normed,
    #             s_normed,
    #             v_sin_normed,
    #             v_cos_normed,
    #             normed_dist_to_goal,
    #             normed_dist_to_ball,
    #             is_possession_team,
    #             is_gk,
    #             is_ball,
    #             goal_sin_normed,
    #             goal_cos_normed,
    #             ball_sin_normed,
    #             ball_cos_normed,
    #             ball_carrier,
    #         ),
    #         axis=-1,
    #     )
    # )

    if graph_features is not None:
        eg = np.ones((X.shape[0], graph_features.shape[0])) * 0.0
        eg[ball_index] = graph_features
        X = np.hstack((X, eg))

    print(X)
    return X
