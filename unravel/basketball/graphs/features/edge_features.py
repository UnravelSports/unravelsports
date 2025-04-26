import numpy as np
from typing import Any


def compute_edge_features(
    x: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise edge features between nodes.

    Args:
        x: NumPy array of shape (n_nodes, n_node_features), typically the node feature matrix.

    Returns:
        e: NumPy array of shape (n_nodes, n_nodes) where e[i, j] is the Euclidean
           distance between feature vectors of node i and node j.
    """
    # Calculate difference between each pair of node feature vectors
    # x[:, None, :] has shape (n, 1, f), x[None, :, :] has shape (1, n, f)
    diff = x[:, None, :] - x[None, :, :]

    # Compute Euclidean norm along the feature axis, resulting in (n, n)
    e = np.linalg.norm(diff, axis=2)
    return e
