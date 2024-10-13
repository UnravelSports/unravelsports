import math
import numpy as np
import inspect

from .utils import (
    unit_vector,
    normalize_angles,
    normalize_distance,
    normalize_coords,
    normalize_speed,
)


def node_features(
    attacking_players,
    defending_players,
    ball,
    max_player_speed,
    max_ball_speed,
    ball_carrier_idx,
    pitch_dimensions,
    include_ball_node: bool = True,
    defending_team_node_value: float = 0.1,
    non_potential_receiver_node_value: float = 0.1,
    function_list=None,
):
    """
    node features matrix is (n_nodes, n_node_features) (<=23, 17)
    each player (and optionally ball) is a node

    player_features n_node_features must be equal to ball_features n_node_features
    """

    goal_mouth_position = (
        pitch_dimensions.pitch_length,
        pitch_dimensions.pitch_width / 2,
    )
    max_dist_to_player = np.sqrt(
        pitch_dimensions.pitch_length**2 + pitch_dimensions.pitch_width**2
    )
    max_dist_to_goal = np.sqrt(
        pitch_dimensions.pitch_length**2 + pitch_dimensions.pitch_width**2
    )

    def player_features(p, team, potential_receiver=None):
        ball_angle = math.atan2(p.y1 - ball.y1, p.x1 - ball.x1)
        goal_angle = math.atan2(
            p.y1 - goal_mouth_position[0], p.x1 - goal_mouth_position[1]
        )
        player_node_features = []
        all_params = {
            "x": p.x1,
            "max_x": pitch_dimensions.x_dim.max,
            "y": p.y1,
            "max_y": pitch_dimensions.y_dim.max,
            "velocity": p.velocity,
            "speed": p.speed,
            "max_speed": max_player_speed,
            "position": p.position,
            "goal_mouth_position": goal_mouth_position,
            "max_dist_to_goal": max_dist_to_goal,
            "goal_angle": goal_angle,
            "ball_position": ball.position,
            "max_dist_to_player": max_dist_to_player,
            "ball_angle": ball_angle,
            "team": team,
            "potential_receiver": potential_receiver,
            "non_potential_receiver_node_value": non_potential_receiver_node_value,
        }
        computed_values = {}
        for func_name, func, reqd_params in function_list:
            try:
                if all(
                    param in all_params for param in reqd_params
                ):  # if all the required parameters exist in all_params, then compute
                    params = [all_params[param] for param in reqd_params]
                    value = func(*params)
                    computed_values[func_name] = value
                    player_node_features.append(value)
                else:  # else, print out the missing parameters. Maybe you should check if there is a default value. Then it is okay if the parameter is not present
                    missing_params = [
                        param for param in reqd_params if param not in all_params
                    ]
                    print(
                        f"Warning: Missing parameters {missing_params} for function '{func_name}'"
                    )
                    computed_values[func_name] = 0
                    player_node_features.append(0)
            except Exception as e:
                print(f"Error while executing function '{func_name}': {e}")
                computed_values[func_name] = None
                player_node_features.append(None)

        return player_node_features

    def ball_features(ball):
        goal_angle = math.atan2(
            ball.y1 - goal_mouth_position[1], ball.x1 - goal_mouth_position[0]
        )
        ball_node_features = []
        all_params = {
            "x": ball.x1,
            "max_x": pitch_dimensions.x_dim.max,
            "y": ball.y1,
            "max_y": pitch_dimensions.y_dim.max,
            "velocity": ball.velocity,
            "speed": ball.speed,
            "max_speed": max_ball_speed,
            "position": ball.position,
            "goal_mouth_position": goal_mouth_position,
            "max_dist_to_goal": max_dist_to_goal,
            "goal_angle": goal_angle,
        }
        computed_values = {}
        for func_name, func, reqd_params in function_list:
            try:
                if all(
                    param in all_params for param in reqd_params
                ):  # if all the required parameters exist in all_params, then compute
                    params = [all_params[param] for param in reqd_params]
                    value = func(*params)
                    computed_values[func_name] = value
                    ball_node_features.append(value)
                else:  # else, print out the missing parameters. Maybe you should check if there is a default value. Then it is okay if the parameter is not present
                    missing_params = [
                        param for param in reqd_params if param not in all_params
                    ]
                    # print(f"Warning: Missing parameters {missing_params} for function '{func_name}'")
                    computed_values[func_name] = 0
                    ball_node_features.append(0)
            except Exception as e:
                print(f"Error while executing function '{func_name}': {e}")
                computed_values[func_name] = None
                ball_node_features.append(None)
        # print(computed_values)
        return np.asarray([ball_node_features])

    # loop over attacking players, grab ball_carrier, potential receiver and intended receiver
    ap_features = np.asarray(
        [
            player_features(p, team=1, potential_receiver=(i != ball_carrier_idx))
            for i, p in enumerate(attacking_players)
        ]
    )

    # loop over defending playres, we don't have ball_carrier, or receivers
    dp_features = np.asarray(
        [
            player_features(p, team=defending_team_node_value)
            for i, p in enumerate(defending_players)
        ]
    )

    # compute ball features
    b_features = ball_features(ball)
    # print(b_features)
    X = np.append(ap_features, dp_features, axis=0)

    if include_ball_node:
        X = np.append(X, b_features, axis=0)

    # convert np.NaN to 0 (zero)
    X = np.nan_to_num(X)
    # print(X)
    # with open('output_file2.txt', 'w') as file:
    #     file.write(np.array2string(X))
    return X
