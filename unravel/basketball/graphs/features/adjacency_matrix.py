import numpy as np
from typing import List, Any


def compute_adjacency_matrix(
    teams: List[Any],
    self_loop: bool = True
) -> np.ndarray:
    """
    Compute the adjacency matrix based on team membership.

    Args:
        teams: List of team identifiers of length n_nodes.
        self_loop: If True, diagonal entries remain 1; if False, zero out self connections.

    Returns:
        A: NumPy array of shape (n_nodes, n_nodes) where A[i, j] = 1.0 if nodes
           i and j belong to the same team, else 0.0. Diagonal set according to self_loop.
    """
    # Convert team list into a NumPy array for vectorized comparisons
    arr = np.array(teams)

    # Create an n x n boolean matrix of team equality
    A = (arr[:, None] == arr[None, :]).astype(float)

    # Optionally remove self-connections by zeroing the diagonal
    if not self_loop:
        np.fill_diagonal(A, 0.0)

    return A
