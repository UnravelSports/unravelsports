import numpy as np
from scipy import sparse
from dataclasses import dataclass


class AdjacenyMatrixConnectType:
    """
    BALL: all players are connected to the ball
    BALL_CARRIER: only ball carrier is connected to the ball
    NO_CONNECTION: no connection between ball and players
    """

    BALL = "ball"
    BALL_CARRIER = "ball_carrier"
    NO_CONNECTION = "no_connection"


class AdjacencyMatrixType:
    """
    DELAUNAY: connect via Delaunay matrix (https://en.wikipedia.org/wiki/Delaunay_triangulation)
    SPLIT_BY_TEAM: connect all players from a team with all players on the same team
    DENSE: connect all players
    DENSE_ATTACKING_PLAYERS: connect only the attacking team
    DENSE_DEFENSIVE_PLAYERS: connect only the defending team
    """

    DELAUNAY = "delaunay"
    SPLIT_BY_TEAM = "split_by_team"
    DENSE = "dense"
    DENSE_ATTACKING_PLAYERS = "dense_ap"
    DENSE_DEFENSIVE_PLAYERS = "dense_dp"


class PredictionLabelType:
    BINARY = "binary"


@dataclass
class Pad:
    max_nodes: int
    max_edges: int
    n_players: int = 11


def normalize_angles(angle):
    old_max = np.pi
    old_min = -np.pi
    new_max = 1
    new_min = 0

    old_range = old_max - old_min
    new_range = new_max - new_min

    return (((angle - old_min) * new_range) / old_range) + new_min


def normalize_between(min_value, max_value, value):
    return (value - min_value) / (max_value - min_value)


def normalize_distance(value, max_distance):
    return value / max_distance


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def normalize_coords(value, max_value):
    return value / max_value


def normalize_sincos(value):
    return (value + 1) / 2


def angle_between(v):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
    # >>> angle_between((1, 0, 0), (0, 1, 0))
    # 1.5707963267948966
    # >>> angle_between((1, 0, 0), (1, 0, 0))
    # 0.0
    # >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1 = v[0:2]
    v2 = v[2:4]
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def non_zeros(A):
    nonzero_idxs = np.where(A == 1)
    nonzero_a = np.count_nonzero(A)
    return nonzero_idxs, nonzero_a


def reindex(m, non_zero_idxs, len_a):
    return m[non_zero_idxs].reshape(len_a, 1)


def make_sparse(a):
    A = sparse.csr_matrix(a)
    return np.nan_to_num(A)


def unit_vector_from_angle(value, angle_radians):
    # Compute velocity components
    value = np.nan_to_num(value, nan=0.0)
    angle_radians = np.nan_to_num(angle_radians, nan=0.0)

    v_x = value * np.cos(angle_radians)
    v_y = value * np.sin(angle_radians)

    # Create velocity vector
    velocity = np.array([v_x, v_y])

    # Normalize the vector (get unit vector)
    norm = np.linalg.norm(velocity)
    if norm == 0:
        return np.zeros_like(velocity)

    return velocity / norm


def normalize_speed(value, max_speed):
    x = value / max_speed
    return np.clip(x, 0, 1)


def normalize_acceleration(value, max_acceleration):
    x = value / max_acceleration
    return np.clip(x, -1, 1)


def normalize_speeds_nfl(s, team, settings):
    ball_mask = team == settings.ball_id
    s_normed = np.zeros_like(s)

    s_normed[ball_mask] = normalize_speed(s[ball_mask], settings.max_ball_speed)

    s_normed[~ball_mask] = normalize_speed(s[~ball_mask], settings.max_player_speed)
    return s_normed


def normalize_speed_differences_nfl(s, team, settings):

    return normalize_speeds_nfl(s, team, settings) * np.sign(s)


def normalize_accelerations_nfl(a, team, settings):
    ball_mask = team == settings.ball_id
    a_normed = np.zeros_like(a)

    a_normed[ball_mask] = normalize_acceleration(
        a[ball_mask], settings.max_ball_acceleration
    )

    a_normed[~ball_mask] = normalize_acceleration(
        a[~ball_mask], settings.max_player_acceleration
    )
    return a_normed


def flatten_to_reshaped_array(arr, s0, s1, as_list=False):
    # Convert the structure into a list of arrays
    flattened_list = [item for sublist in arr for item in sublist]
    # Concatenate the arrays into one single array
    result_array = np.concatenate(flattened_list).reshape(s0, s1)
    return result_array if not as_list else result_array.tolist()
