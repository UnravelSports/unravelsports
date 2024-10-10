import numpy as np

from ....utils import AdjacencyMatrixType, AdjacenyMatrixConnectType


def compute_adjacency_matrix(team, possession_team, settings):
    adjacency_matrix_type = settings.adjacency_matrix_type
    adjacency_matrix_connect_type = settings.adjacency_matrix_connect_type
    ball_id = settings.ball_id

    exclusion_ids = np.asarray([ball_id, *np.unique(possession_team)])
    defensive_team = np.setdiff1d(team, exclusion_ids)[0]
    if adjacency_matrix_type == AdjacencyMatrixType.DENSE:
        adjacency_matrix = np.ones((team.shape[0], team.shape[0])).astype(np.int32)
    elif adjacency_matrix_type == AdjacencyMatrixType.DENSE_ATTACKING_PLAYERS:
        is_att = team == np.unique(possession_team)[0]
        adjacency_matrix = np.outer(is_att, is_att).astype(int)
    elif adjacency_matrix_type == AdjacencyMatrixType.DENSE_DEFENSIVE_PLAYERS:
        is_def = team == defensive_team
        adjacency_matrix = np.outer(is_def, is_def).astype(int)
    elif adjacency_matrix_type == AdjacencyMatrixType.SPLIT_BY_TEAM:
        # Create a pairwise team comparison matrix
        adjacency_matrix = np.equal(team[:, None], team[None, :]).astype(np.int32)

    if adjacency_matrix_connect_type:
        if adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL:
            # Create a mask where either team is "football"
            football_mask = (team[:, None] == ball_id) | (team[None, :] == ball_id)
            # Set entries to 1 where either team is "football"
            adjacency_matrix = np.where(football_mask, 1, adjacency_matrix)
        elif adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL_CARRIER:
            raise NotImplementedError(
                "No ball carrier information exists in the BigDataBowl dataset, please choose a different AdjacenyMatrixConnectType..."
            )

    return adjacency_matrix


def delaunay_adjacency_matrix():
    pass
