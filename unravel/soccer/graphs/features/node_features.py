import math
import numpy as np

from ....utils import (
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

        player_node_features = [
            (
                0.0
                if np.isnan(p.x1)
                else normalize_coords(p.x1, pitch_dimensions.x_dim.max)
            ),
            (
                0.0
                if np.isnan(p.y1)
                else normalize_coords(p.y1, pitch_dimensions.y_dim.max)
            ),
            0.0 if np.isnan(p.x1) else unit_vector(p.velocity)[0],
            0.0 if np.isnan(p.x1) else unit_vector(p.velocity)[1],
            (
                0.0
                if np.isnan(p.x1)
                else round(normalize_speed(p.speed, max_speed=max_player_speed), 3)
            ),
            (
                0.0
                if np.isnan(p.x1)
                else normalize_angles(np.arctan2(p.velocity[1], p.velocity[0]))
            ),
            (
                0.0
                if np.isnan(p.x1)
                else normalize_distance(
                    np.linalg.norm(p.position - goal_mouth_position),
                    max_distance=max_dist_to_goal,
                )
            ),  # distance to the goal mouth
            0.0 if np.isnan(p.x1) else normalize_angles(goal_angle),
            (
                0.0
                if np.isnan(p.x1)
                else normalize_distance(
                    np.linalg.norm(p.position - ball.position),
                    max_distance=max_dist_to_player,
                )
            ),  # distance to the ball
            0.0 if np.isnan(p.x1) else normalize_angles(ball_angle),
            0.0 if np.isnan(p.x1) else team,
            # 1 if player is on same team but not in possession, 0.1 for all other players, 0.1 if the player is 'missing'
            (
                0.0
                if np.isnan(p.x1)
                else 1.0 if potential_receiver else non_potential_receiver_node_value
            ),
        ]
        return player_node_features

    def ball_features(ball):
        goal_angle = math.atan2(
            ball.y1 - goal_mouth_position[1], ball.x1 - goal_mouth_position[0]
        )
        ball_node_features = [
            normalize_coords(ball.x1, pitch_dimensions.x_dim.max),
            normalize_coords(ball.y1, pitch_dimensions.y_dim.max),
            unit_vector(ball.velocity)[0],
            unit_vector(ball.velocity)[1],
            round(normalize_speed(ball.speed, max_speed=max_ball_speed), 3),
            normalize_angles(np.arctan2(ball.velocity[1], ball.velocity[0])),
            normalize_distance(
                np.linalg.norm(ball.position - goal_mouth_position),
                max_distance=max_dist_to_goal,
            ),  # distance to the goal mouth
            normalize_angles(goal_angle),
            # ball_angle 2x, ball_dist 2x, attacking_team 2x, ball carrier, potential receiver (all always 0 for ball)
            0,
            0,
            0,
            0,  # , 0
        ]

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
    X = np.append(ap_features, dp_features, axis=0)

    if include_ball_node:
        X = np.append(X, b_features, axis=0)

    # convert np.NaN to 0 (zero)
    X = np.nan_to_num(X)
    return X
