import numpy as np


from ....utils import (
    normalize_coords,
    normalize_speeds_nfl,
    normalize_angles,
    normalize_distance,
    unit_vector_from_angle,
    normalize_speeds_nfl,
    normalize_accelerations_nfl,
)


def compute_node_features(
    x, y, s, a, o, dir, team, official_position, possession_team, settings
):
    ball_id = settings.ball_id

    goal_mouth_position = (
        settings.pitch_dimensions.pitch_length,
        settings.pitch_dimensions.pitch_width / 2,
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
        ball_position = np.sarray([np.nan, np.nan])

    x_normed = normalize_coords(x, settings.pitch_dimensions.x_dim.max)
    y_normed = normalize_coords(y, settings.pitch_dimensions.y_dim.max)
    uv_sa = unit_vector_from_angle(value=s, angle_radians=dir)
    s_normed = normalize_speeds_nfl(s, team, ball_id, settings)
    uv_aa = unit_vector_from_angle(value=a, angle_radians=dir)
    a_normed = normalize_accelerations_nfl(a, team, ball_id, settings)

    dir_normed = normalize_angles(dir)
    o_normed = normalize_angles(o)

    dist_to_goal = np.linalg.norm(position - goal_mouth_position, axis=1)
    normed_dist_to_goal = normalize_distance(
        value=dist_to_goal, max_distance=max_dist_to_goal
    )

    dist_to_ball = np.linalg.norm(position - ball_position, axis=1)
    normed_dist_to_ball = normalize_distance(
        value=dist_to_ball, max_distance=max_dist_to_player
    )

    dist_to_end_zone = y - settings.pitch_dimensions.end_zone
    normed_dist_to_end_zone = dist_to_end_zone / settings.pitch_dimensions.y_dim.max

    is_possession_team = np.where(team == possession_team, 1, 0)
    is_qb = np.where(official_position == settings.qb_id, 1, 0)
    is_ball = np.where(team == ball_id, 1, 0)

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
                dir_normed,
                o_normed,
                normed_dist_to_goal,
                normed_dist_to_ball,
                normed_dist_to_end_zone,
                is_possession_team,
                is_qb,
                is_ball,
            ),
            axis=-1,
        )
    )

    return X
