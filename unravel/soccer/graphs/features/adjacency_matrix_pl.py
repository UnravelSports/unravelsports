import numpy as np
from scipy.spatial import Delaunay


from ....utils import AdjacencyMatrixType, AdjacenyMatrixConnectType
from ...dataset.kloppy_polars import Constant


def compute_adjacency_matrix(settings, **kwargs):
    team = kwargs["team_id"]
    ball_owning_team = kwargs["ball_owning_team_id"]
    ball_carrier_idx = kwargs["ball_carrier_idx"]

    adjacency_matrix_type = settings.adjacency_matrix_type
    adjacency_matrix_connect_type = settings.adjacency_matrix_connect_type
    ball_id = Constant.BALL

    exclusion_ids = np.asarray([ball_id, *np.unique(ball_owning_team)])

    defensive_team = np.setdiff1d(team, exclusion_ids)[0]
    if adjacency_matrix_type == AdjacencyMatrixType.DENSE:
        adjacency_matrix = np.ones((team.shape[0], team.shape[0])).astype(np.int32)
    elif adjacency_matrix_type == AdjacencyMatrixType.DENSE_AP:
        is_att = team == np.unique(ball_owning_team)[0]
        adjacency_matrix = np.outer(is_att, is_att).astype(int)
    elif adjacency_matrix_type == AdjacencyMatrixType.DENSE_DP:
        is_def = team == defensive_team
        adjacency_matrix = np.outer(is_def, is_def).astype(int)
    elif adjacency_matrix_type == AdjacencyMatrixType.SPLIT_BY_TEAM:
        # Create a pairwise team comparison matrix
        adjacency_matrix = np.equal(team[:, None], team[None, :]).astype(np.int32)
    elif adjacency_matrix_type == AdjacencyMatrixType.DELAUNAY:
        raise NotImplementedError("Delaunay matrix not implemented for Soccer...")
    else:
        raise NotImplementedError("Please specify an existing AdjacencyMatrixType...")

    if adjacency_matrix_connect_type:
        # Create a mask where either team is "ball"
        ball_mask = (team[:, None] == ball_id) | (team[None, :] == ball_id)
        if adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL:
            # Set entries to 1 where either team is "ball"
            adjacency_matrix = np.where(ball_mask, 1, adjacency_matrix)
        elif adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL_CARRIER:
            if ball_carrier_idx is not None:
                adjacency_matrix[ball_carrier_idx, ball_mask[ball_carrier_idx, :]] = 1
                adjacency_matrix[ball_mask[:, ball_carrier_idx], ball_carrier_idx] = 1

    return adjacency_matrix
