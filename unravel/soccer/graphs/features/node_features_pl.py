import numpy as np


def add_global_features(node_features, global_features, global_feature_type, **kwargs):
    if global_feature_type == "ball":
        eg = np.ones((node_features.shape[0], global_features.shape[0])) * 0.0
        eg[kwargs["ball_idx"]] = global_features
        node_features = np.hstack((node_features, eg))
    elif global_feature_type == "all":
        eg = np.tile(global_features, (node_features.shape[0], 1))
        node_features = np.hstack((node_features, eg))
    else:
        raise ValueError("global_features_type should be either of {ball, all}")
    return node_features


def compute_node_features(
    funcs,
    opts,
    settings,
    **kwargs,
):
    """
    Parameters:
    - funcs: List of node feature functions
    - opts: Dictionary with additional options
    - settings: The settings object containing pitch dimensions
    - **kwargs: Dictionary of numpy arrays coming from `exprs_variables`.

    Returns:
    - X: Computed node features
    """
    reference_shape = kwargs["team_id"].shape

    if opts is not None:
        combined_opts = {**kwargs, **opts}
    else:
        combined_opts = {**kwargs}

    if "settings" in combined_opts:
        raise ValueError(
            "settings is a reserved keyword in the node feature options, it should not be in 'kwargs' and 'node_feature_opts'"
        )

    combined_opts.update({"settings": settings})
    node_feature_values = []

    for func in funcs:
        try:
            value = func(**combined_opts)

            # Ensure value is a numpy array
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Function {func.__name__} must return a numpy array")

            # Handle different shapes
            if value.shape == reference_shape:
                # Single column case
                node_feature_values.append(value)
            elif value.shape[0] == reference_shape[0] and len(value.shape) > 1:
                # Multi-column case - split and append each column
                node_feature_values.extend([value[:, i] for i in range(value.shape[1])])
            else:
                raise ValueError(
                    f"Shape mismatch: expected first dimension to be {reference_shape[0]}, got {value.shape}"
                )

        except Exception as e:
            # Include useful context in error
            import inspect

            func_str = inspect.getsource(func).strip()
            error_msg = (
                f"Error processing node feature function:\n"
                f"{func.__name__} defined as:\n"
                f"{func_str}\n\n"
                f"Error: {str(e)}\n"
                f"Expected shape: ({reference_shape[0]}, k) (try np.column_stack) or {reference_shape}"
            )
            raise ValueError(error_msg) from e

    X = np.nan_to_num(
        np.stack(
            node_feature_values,
            axis=-1,
        )
    )
    return X
