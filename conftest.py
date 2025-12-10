import pytest
import sys


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "spektral: tests that require spektral (Python 3.11 only)"
    )
    config.addinivalue_line("markers", "torch: tests that require PyTorch")


def pytest_collection_modifyitems(config, items):
    """Automatically skip tests based on available dependencies"""

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

    skip_spektral = pytest.mark.skip(reason="Spektral not installed")
    skip_torch = pytest.mark.skip(reason="PyTorch/PyG not installed")

    for item in items:
        if "spektral" in item.keywords and not has_spektral:
            item.add_marker(skip_spektral)
        if "torch" in item.keywords and not has_torch:
            item.add_marker(skip_torch)
