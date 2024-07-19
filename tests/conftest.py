import sys
import os

from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def base_dir() -> Path:
    return Path(__file__).parent
