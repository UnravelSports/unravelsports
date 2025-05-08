import numpy as np

from ....utils import (
    normalize_distance,
    normalize_speed,
    normalize_sincos,
    normalize_speed_differences,
    angle_between,
    non_zeros,
    reindex,
)

import numpy as np

from ...dataset.kloppy_polars import Constant


def compute_edge_features(adjacency_matrix, funcs, opts, settings, **kwargs):
    reference_shape = (kwargs["team_id"].shape[0], kwargs["team_id"].shape[0])

    if opts is not None:
        combined_opts = {**kwargs, **opts}
    else:
        combined_opts = {**kwargs}

    if "settings" in combined_opts:
        raise ValueError(
            "settings is a reserved keyword in the edge feature options, it should not be in 'kwargs' and 'edge_feature_opts'"
        )

    combined_opts.update({"settings": settings})
    edge_feature_values = []

    for func in funcs:
        try:
            value = func(**combined_opts)
            if isinstance(value, tuple):
                for m in value:
                    if m.shape == reference_shape:
                        edge_feature_values.append(m)
                    else:
                        raise ValueError(
                            f"Shape mismatch: expected first dimension to be {reference_shape[0]}, got {value.shape}"
                        )
            elif isinstance(value, np.ndarray):
                if value.shape == reference_shape:
                    edge_feature_values.append(value)
                else:
                    raise ValueError(
                        f"Shape mismatch: expected first dimension to be {reference_shape[0]}, got {value.shape}"
                    )
            else:
                raise ValueError(
                    f"Function {func.__name__} must return a numpy array of shape (N, N) or a tuple with multiple arrays of shape (N, N)"
                )

        except Exception as e:
            import inspect

            func_str = inspect.getsource(func).strip()
            error_msg = (
                f"Error processing edge feature function:\n"
                f"{func.__name__} defined as:\n"
                f"{func_str}\n\n"
                f"Error: {str(e)}\n"
                f"Expected shape: {reference_shape}, i.e. (N, N)"
            )
            raise ValueError(error_msg) from e

    non_zero_idxs, len_a = non_zeros(A=adjacency_matrix)

    e_list = list(
        [reindex(matrix, non_zero_idxs, len_a) for matrix in edge_feature_values]
    )
    e = np.concatenate(e_list, axis=1)
    return np.nan_to_num(e)
