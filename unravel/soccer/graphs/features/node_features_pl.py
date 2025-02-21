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

    feature_func_map = {
        'x': {'func': lambda value: value, 'defaults': {'value': x}},
        'x_normed': {'func': normalize_between, 'defaults': {'value': x, 'max_value': settings.pitch_dimensions.x_dim.max, 'min_value': settings.pitch_dimensions.x_dim.min}},
        'y': {'func': lambda value: value, 'defaults': {'value': y}},
        'y_normed': {'func': normalize_between, 'defaults': {'value': y, 'max_value': settings.pitch_dimensions.y_dim.max, 'min_value': settings.pitch_dimensions.y_dim.min}},
        's': {'func': lambda value: value, 'defaults': {'value': s}},
        's_normed': {'func': normalize_speeds_nfl, 'defaults': {'s': s, 'team': team, 'ball_id': ball_id, 'settings': settings}},
        'velocity': {'func': lambda value: value, 'defaults': {'value': velocity}},
        
        '''
            uv_velocity = unit_vectors(velocity)
            angles = normalize_angles(np.arctan2(uv_velocity[:, 1], uv_velocity[:, 0]))
            
            v_sin_normed = normalize_sincos(np.sin(angles))
            v_cos_normed = normalize_sincos(np.cos(angles))
        '''
        
        'v_sin_normed': {'func': normalize_sincos, 'defaults': {'value': np.sin(normalize_angles(np.arctan2(unit_vectors(velocity)[:, 1],  unit_vectors(velocity)[:, 0])))}},
        'v_cos_normed': {'func': normalize_sincos, 'defaults': {'value': np.cos(normalize_angles(np.arctan2(unit_vectors(velocity)[:, 1],  unit_vectors(velocity)[:, 0])))}},
        'dist_to_goal': {'func': lambda value: value, 'defaults': {'value': np.linalg.norm(position - goal_mouth_position, axis=1)}},
        'normed_dist_to_goal': {'func': normalize_distance, 'defaults': {'value': np.linalg.norm(position - goal_mouth_position, axis=1), 'max_distance': max_dist_to_goal}},
        'normed_dist_to_ball': {'func': normalize_distance, 'defaults': {'value': dist_to_ball, 'max_distance': max_dist_to_player}},
        'vec_to_goal': {'func': lambda value: value, 'defaults': {'value': goal_mouth_position - position}},
        'angle_to_goal': {'func': lambda value: value, 'defaults': {'value': np.arctan2((goal_mouth_position - position)[:, 1], (goal_mouth_position - position)[:, 0])}},
        'goal_sin_normed': {'func': normalize_sincos, 'defaults': {'value': np.sin(np.arctan2((goal_mouth_position - position)[:, 1], (goal_mouth_position - position)[:, 0]))}},
        'goal_cos_normed': {'func': normalize_sincos, 'defaults': {'value': np.cos(np.arctan2((goal_mouth_position - position)[:, 1], (goal_mouth_position - position)[:, 0]))}},
        'vec_to_ball': {'func': lambda value: value, 'defaults': {'value': ball_position - position}},
        'angle_to_ball': {'func': lambda value: value, 'defaults': {'value': np.arctan2((ball_position - position)[:, 1], (ball_position - position)[:, 0])}},
        'ball_sin_normed': {'func': normalize_sincos, 'defaults': {'value': np.sin(np.arctan2((ball_position - position)[:, 1], (ball_position - position)[:, 0]))}},
        'ball_cos_normed': {'func': normalize_sincos, 'defaults': {'value': np.cos(np.arctan2((ball_position - position)[:, 1], (ball_position - position)[:, 0]))}},
        'ball_carrier': {'func': lambda value: value, 'defaults': {'value': ball_carrier}},
        'is_possession_team': {'func': lambda value: value, 'defaults': {'value': np.where(team == possession_team, 1, settings.defending_team_node_value)}},
        'is_ball': {'func': lambda value: value, 'defaults': {'value': np.where(team == ball_id, 1, 0)}},
        'is_gk': {'func': lambda value: value, 'defaults': {'value': is_gk}},
    }
    
    computed_features = []
    
    for feature, custom_params in feature_dict.items():
        if feature in feature_func_map:
            params = feature_func_map[feature]['defaults'].copy()
            params.update(custom_params)
            computed_features.append(feature_func_map[feature]['func'](**params))
    
    X = np.nan_to_num(np.stack(computed_features, axis=-1))
    
    if graph_features is not None:
        eg = np.ones((X.shape[0], graph_features.shape[0])) * 0.0
        eg[ball_index] = graph_features
        X = np.hstack((X, eg))

    #print(X)
    return X
