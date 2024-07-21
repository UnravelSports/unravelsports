import numpy as np
from scipy.spatial import Delaunay


from .utils import AdjacencyMatrixType, AdjacenyMatrixConnectType


def delaunay_adjacency_matrix(
    attacking_players,
    defending_players,
    adjacency_matrix_connect_type: AdjacenyMatrixConnectType,
    ball_carrier_idx: int = None,
    self_loop_ball: bool = False,
):
    """
    Computes the Delaunay triangulation of the given points
    :param x:np.asarray of shape (num_nodes, 2)
    :return: the computed adjacency matrix
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ("No such package `scipy`... Please install using `pip install scipy`")
    # ! att_players needs to always go first here, because _ball_carrier_idx is associated with att_players
    pts = np.asarray([p.position for p in attacking_players + defending_players])

    tri = Delaunay(pts)
    edges_explicit = np.concatenate(
        (tri.vertices[:, :2], tri.vertices[:, 1:], tri.vertices[:, ::2]), axis=0
    )
    adj = np.zeros((pts.shape[0], pts.shape[0]))
    adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.0
    A = np.clip(adj + adj.T, 0, 1)

    num_p = len(pts)

    if adjacency_matrix_connect_type:
        if adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL:
            Z1 = np.ones((num_p, 1), dtype=int)  # Create off-diagonal ones array
            Z2 = np.ones((1, num_p), dtype=int)  # Create off-diagonal ones array
        elif adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL_CARRIER:
            Z1 = np.zeros((num_p, 1), dtype=int)  # Create off-diagonal ones array
            Z2 = np.zeros((1, num_p), dtype=int)  # Create off-diagonal ones array
            if not ball_carrier_idx:
                # if we don't have a ball carrier, because ball is too far away from furthest player
                # then set ball carrier index to 0
                Z1[ball_carrier_idx, :] = 0
                Z2[:, ball_carrier_idx] = 0
            else:
                Z1[ball_carrier_idx, :] = 1
                Z2[:, ball_carrier_idx] = 1
        else:
            Z1 = np.zeros((num_p, 1), dtype=int)  # Create off-diagonal zeros array
            Z2 = np.zeros((1, num_p), dtype=int)  # Create off-diagonal zeros array

        ball_connect_val = 1 if self_loop_ball else 0
        b = np.asarray([[ball_connect_val]])
        A = np.asarray(np.bmat([[A, Z1], [Z2, b]]))
    return A


def adjacency_matrix(
    attacking_players,
    defending_players,
    adjacency_matrix_connect_type: AdjacenyMatrixConnectType,
    adjacency_matrix_type: AdjacencyMatrixType,
    ball_carrier_idx: int = None,
):
    """
    This adjacency matrix adjusts for teams with less than 10 players
    """
    # ! att_players needs to always go first here, because _ball_carrier_idx is associated with att_players
    ap = len(attacking_players)
    ap_ones = np.ones(shape=(ap, ap))

    dp = len(defending_players)
    dp_ones = np.ones(shape=(dp, dp))

    if adjacency_matrix_type == AdjacencyMatrixType.DENSE:
        Z1 = np.ones((ap, dp), dtype=int)  # Create off-diagonal ones array
        Z2 = np.ones((dp, ap), dtype=int)  # Create off-diagonal ones array
    elif adjacency_matrix_type == AdjacencyMatrixType.DENSE_ATTACKING_PLAYERS:
        Z1 = np.ones((ap, dp), dtype=int)  # Create off-diagonal ones array
        Z2 = np.zeros((dp, ap), dtype=int)  # Create off-diagonal zeros array
    elif adjacency_matrix_type == AdjacencyMatrixType.DENSE_DEFENSIVE_PLAYERS:
        Z1 = np.zeros((ap, dp), dtype=int)  # Create off-diagonal zeros array
        Z2 = np.ones((dp, ap), dtype=int)  # Create off-diagonal ones array
    elif adjacency_matrix_type == AdjacencyMatrixType.SPLIT_BY_TEAM:
        Z1 = np.zeros((ap, dp), dtype=int)  # Create off-diagonal zeros array
        Z2 = np.zeros((dp, ap), dtype=int)  # Create off-diagonal zeros array

    A = np.asarray(np.bmat([[ap_ones, Z1], [Z2, dp_ones]]))

    if adjacency_matrix_connect_type:
        if adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL:
            Z1 = np.ones((ap + dp, 1), dtype=int)  # Create off-diagonal ones array
            Z2 = np.ones((1, dp + ap), dtype=int)  # Create off-diagonal ones array
        elif adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL_CARRIER:
            Z1 = np.zeros((ap + dp, 1), dtype=int)  # Create off-diagonal ones array
            Z2 = np.zeros((1, ap + dp), dtype=int)  # Create off-diagonal ones array
            if ball_carrier_idx:
                Z1[ball_carrier_idx, :] = 1
                Z2[:, ball_carrier_idx] = 1
        else:
            Z1 = np.zeros((ap + dp, 1), dtype=int)  # Create off-diagonal zeros array
            Z2 = np.zeros((1, dp + ap), dtype=int)  # Create off-diagonal zeros array

        b = np.asarray([[1]])
        A = np.asarray(np.bmat([[A, Z1], [Z2, b]]))

    return A


def delaunay_adjacency_matrix(
    attacking_players,
    defending_players,
    adjacency_matrix_connect_type: AdjacenyMatrixConnectType,
    ball_carrier_idx: int = None,
    self_loop_ball: bool = False,
):
    """
    Computes the Delaunay triangulation of the given points
    :param x:np.asarray of shape (num_nodes, 2)
    :return: the computed adjacency matrix
    """
    # ! att_players needs to always go first here, because _ball_carrier_idx is associated with att_players
    pts = np.asarray([p.position for p in attacking_players + defending_players])

    # Create a mask for valid positions (non-NaN)
    valid_mask = ~np.isnan(pts).any(axis=1)

    # Filter positions to include only valid positions
    valid_positions = pts[valid_mask]

    # Perform Delaunay triangulation on valid positions
    tri = Delaunay(valid_positions)

    # Create edges from the triangulation
    edges_explicit = np.concatenate(
        (tri.simplices[:, :2], tri.simplices[:, 1:], tri.simplices[:, ::2]), axis=0
    )

    # Initialize the adjacency matrix for valid positions
    adj_valid = np.zeros((valid_positions.shape[0], valid_positions.shape[0]))
    adj_valid[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.0
    A_valid = np.clip(adj_valid + adj_valid.T, 0, 1)

    # Initialize the full adjacency matrix with the original size
    A = np.zeros((pts.shape[0], pts.shape[0]))

    # Map the valid adjacency matrix back to the original indices
    valid_indices = np.where(valid_mask)[0]
    A[np.ix_(valid_indices, valid_indices)] = A_valid

    num_p = len(pts)

    if adjacency_matrix_connect_type:
        if adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL:
            Z1 = np.ones((num_p, 1), dtype=int)  # Create off-diagonal ones array
            Z2 = np.ones((1, num_p), dtype=int)  # Create off-diagonal ones array
        elif adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL_CARRIER:
            Z1 = np.zeros((num_p, 1), dtype=int)  # Create off-diagonal ones array
            Z2 = np.zeros((1, num_p), dtype=int)  # Create off-diagonal ones array
            if not ball_carrier_idx:
                # if we don't have a ball carrier, because ball is too far away from furthest player
                # then set ball carrier index to 0
                Z1[ball_carrier_idx, :] = 0
                Z2[:, ball_carrier_idx] = 0
            else:
                Z1[ball_carrier_idx, :] = 1
                Z2[:, ball_carrier_idx] = 1
        else:
            Z1 = np.zeros((num_p, 1), dtype=int)  # Create off-diagonal zeros array
            Z2 = np.zeros((1, num_p), dtype=int)  # Create off-diagonal zeros array

        ball_connect_val = 1 if self_loop_ball else 0
        b = np.asarray([[ball_connect_val]])
        A = np.asarray(np.bmat([[A, Z1], [Z2, b]]))
    return A
