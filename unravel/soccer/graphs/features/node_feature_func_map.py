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
from typing import TypedDict, Dict, Optional, Union

class NodeFeatureDefaults(TypedDict):
    #value: Optional[Union[float, np.ndarray]]
    value: float
    max_value: Optional[float] 
    min_value: Optional[float] 
    team: Optional[int]  
    ball_id: Optional[int] 
    s: Optional[Union[float, np.ndarray]]
    goal_mouth_position: Optional[np.ndarray]
    ball_position: Optional[np.ndarray]
    is_gk: Optional[bool]

class FeatureFuncMap(TypedDict):
    defaults: NodeFeatureDefaults

def get_node_feature_func_map(x=None,
    y=None,
    s=None,
    velocity=None,
    team=None,
    possession_team=None,
    is_gk=None,
    ball_carrier=None,
    graph_features=None,
    settings=None):
    
    ball_id = Constant.BALL

    position = np.stack((x, y), axis=-1) if x and y else None

    if team and len(np.where(team == ball_id)) >= 1:
        ball_index = np.where(team == ball_id)[0]
        ball_position = position[ball_index][0]
    else:
        ball_position = np.asarray([np.nan, np.nan])
        ball_index = 0

    goal_mouth_position = (
        settings.pitch_dimensions.x_dim.max,
        (settings.pitch_dimensions.y_dim.max + settings.pitch_dimensions.y_dim.min) / 2,
    ) if settings else None
    
    max_dist_to_player = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    ) if settings else None
    
    max_dist_to_goal = np.sqrt(
        settings.pitch_dimensions.pitch_length**2
        + settings.pitch_dimensions.pitch_width**2
    ) if settings else None

    position, ball_position, dist_to_ball = distance_to_ball(
        x=x, y=y, team=team, ball_id=ball_id
    ) if all([x,y,team]) else (None,None,None)
    
    feature_func_map: Dict[str, FeatureFuncMap] = {
        "x": {"func": lambda value: value, "defaults": {"value": x}},
        "x_normed": {
            "func": normalize_between,
            "defaults": {
                "value": x if x else None,
                "max_value": settings.pitch_dimensions.x_dim.max if settings else None,
                "min_value": settings.pitch_dimensions.x_dim.min if settings else None,
            },
        },
        "y": {"func": lambda value: value, "defaults": {"value": y}},
        "y_normed": {
            "func": normalize_between,
            "defaults": {
                "value": y if y else None,
                "max_value": settings.pitch_dimensions.y_dim.max if settings else None,
                "min_value": settings.pitch_dimensions.y_dim.min if settings else None,
            },
        },
        "s": {"func": lambda value: value, "defaults": {"value": s}},
        "s_normed": {
            "func": normalize_speeds_nfl,
            "defaults": {
                "s": s if s else None,
                "team": team if team else None,
                "ball_id": ball_id,
                "settings": settings if settings else None,
            },
        },
        "velocity": {"func": lambda value: value, "defaults": {"value": velocity if velocity else None}},
        # """
        #     uv_velocity = unit_vectors(velocity)
        #     angles = normalize_angles(np.arctan2(uv_velocity[:, 1], uv_velocity[:, 0]))
        #     v_sin_normed = normalize_sincos(np.sin(angles))
        #     v_cos_normed = normalize_sincos(np.cos(angles))
        # """
        "v_sin_normed": {
            "func": normalize_sincos,
            "defaults": {
                "value": np.sin(
                    normalize_angles(
                        np.arctan2(
                            unit_vectors(velocity)[:, 1], unit_vectors(velocity)[:, 0]
                        )
                    )
                ) if velocity else None
            },
        },
        "v_cos_normed": {
            "func": normalize_sincos,
            "defaults": {
                "value": np.cos(
                    normalize_angles(
                        np.arctan2(
                            unit_vectors(velocity)[:, 1], unit_vectors(velocity)[:, 0]
                        )
                    )
                ) if velocity else None
            },
        },
        "dist_to_goal": {
            "func": lambda value: value,
            "defaults": {
                "value": np.linalg.norm(position - goal_mouth_position, axis=1) if position and goal_mouth_position else None
            },
        },
        "normed_dist_to_goal": {
            "func": normalize_distance,
            "defaults": {
                "value": np.linalg.norm(position - goal_mouth_position, axis=1) if position and goal_mouth_position else None,
                "max_distance": max_dist_to_goal,
            },
        },
        "normed_dist_to_ball": {
            "func": normalize_distance,
            "defaults": {"value": dist_to_ball, "max_distance": max_dist_to_player},
        },
        "vec_to_goal": {
            "func": lambda value: value,
            "defaults": {"value": goal_mouth_position - position if goal_mouth_position and position else None},
        },
        "angle_to_goal": {
            "func": lambda value: value,
            "defaults": {
                "value": np.arctan2(
                    (goal_mouth_position - position)[:, 1],
                    (goal_mouth_position - position)[:, 0],
                ) if goal_mouth_position and position else None
            },
        },
        "goal_sin_normed": {
            "func": normalize_sincos,
            "defaults": {
                "value": np.sin(
                    np.arctan2(
                        (goal_mouth_position - position)[:, 1],
                        (goal_mouth_position - position)[:, 0],
                    )
                ) if goal_mouth_position and position else None
            },
        },
        "goal_cos_normed": {
            "func": normalize_sincos,
            "defaults": {
                "value": np.cos(
                    np.arctan2(
                        (goal_mouth_position - position)[:, 1],
                        (goal_mouth_position - position)[:, 0],
                    )
                ) if goal_mouth_position and position else None
            },
        },
        "vec_to_ball": {
            "func": lambda value: value,
            "defaults": {"value": ball_position - position if ball_position and position else None},
        },
        "angle_to_ball": {
            "func": lambda value: value,
            "defaults": {
                "value": np.arctan2(
                    (ball_position - position)[:, 1], (ball_position - position)[:, 0]
                ) if ball_position and position else None
            },
        },
        "ball_sin_normed": {
            "func": normalize_sincos,
            "defaults": {
                "value": np.sin(
                    np.arctan2(
                        (ball_position - position)[:, 1],
                        (ball_position - position)[:, 0],
                    )
                ) if ball_position and position else None
            },
        },
        "ball_cos_normed": {
            "func": normalize_sincos,
            "defaults": {
                "value": np.cos(
                    np.arctan2(
                        (ball_position - position)[:, 1],
                        (ball_position - position)[:, 0],
                    )
                ) if ball_position and position else None
            },
        },
        "ball_carrier": {
            "func": lambda value: value,
            "defaults": {"value": ball_carrier},
        },
        "is_possession_team": {
            "func": lambda value: value,
            "defaults": {
                "value": np.where(
                    team == possession_team, 1, settings.defending_team_node_value
                ) if all([team, possession_team, settings]) else None
            },
        },
        "is_ball": {
            "func": lambda value: value,
            "defaults": {"value": np.where(team == ball_id, 1, 0)} if team else None,
        },
        "is_gk": {"func": lambda value: value, "defaults": {"value": is_gk}},
    }

    return feature_func_map