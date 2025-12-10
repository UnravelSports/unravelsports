import pytest
import sys
import os


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "spektral: tests that require spektral (Python 3.11 only)"
    )
    config.addinivalue_line("markers", "torch: tests that require PyTorch")
    config.addinivalue_line(
        "markers", "local_only: tests that should only run in local environment"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on available dependencies and environment"""

    try:
        import spektral

        has_spektral = True
    except ImportError:
        has_spektral = False

    try:
        import torch
        import torch_geometric

        has_torch = True
    except ImportError:
        has_torch = False

    # Check if running in CI or non-local environment
    is_ci = os.getenv("CI") is not None

    skip_spektral = pytest.mark.skip(reason="Spektral not installed")
    skip_torch = pytest.mark.skip(reason="PyTorch/PyG not installed")
    skip_local = pytest.mark.skip(
        reason="Skipping local-only tests in CI/non-local environment"
    )

    for item in items:
        if "spektral" in item.keywords and not has_spektral:
            item.add_marker(skip_spektral)
        if "torch" in item.keywords and not has_torch:
            item.add_marker(skip_torch)
        if "local_only" in item.keywords and is_ci:
            item.add_marker(skip_local)
