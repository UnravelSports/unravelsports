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


def normalize_distance(value, max_distance):
    return value / max_distance


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm


def normalize_coords(value, max_value):
    return value / max_value


def normalize_speed(value, max_speed):
    x = value / max_speed
    try:
        return 1 if x > 1 else 0 if x < 0 else x
    except ValueError:
        x[x < 0] = 0
        x[x > 1] = 1
        return x


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
