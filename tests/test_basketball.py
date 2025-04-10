import os
import json
import tempfile
import numpy as np
import polars as pl
import pytest

# Import the Basketball classes from the basketball module
from unravel.basketball.dataset.dataset import BasketballDataset
from unravel.basketball.graphs.graph_settings import BasketballGraphSettings
from unravel.basketball.graphs.pitch_dimensions import BasketballPitchDimensions
from unravel.basketball.graphs.graph_converter import BasketballGraphConverter

@pytest.fixture
def sample_basketball_json(tmp_path):
    """
    Creates a temporary JSON file with sample NBA tracking data.
    The data structure contains:
      - A game identifier ("NBA001")
      - A list of events with one moment that includes two entities.
    """
    sample_data = {
        "gameid": "NBA001",
        "events": [
            {
                "moments": [
                    # Moment structure: [quarter, <unused>, game_clock, shot_clock, <unused>, list of entities]
                    [1, None, "12.5", "24", None, [
                        ["Lakers", "LeBron", 50.0, 25.0],
                        ["Celtics", "Tatum", 30.0, 20.0]
                    ]]
                ]
            }
        ]
    }
    file_path = tmp_path / "sample.json"
    file_path.write_text(json.dumps(sample_data))
    return file_path

def test_dataset_load_local(sample_basketball_json):
    """
    Test BasketballDataset: load a local file and verify that the DataFrame is built correctly.
    """
    dataset = BasketballDataset(str(sample_basketball_json))
    df = dataset.load()
    expected_columns = {"game_id", "event_id", "frame_id", "quarter", "game_clock", "shot_clock", "team", "player", "x", "y"}
    assert set(df.columns) >= expected_columns
    # Expect two rows (one for each player in the moment)
    assert df.height == 2
    # Verify that game_id matches "NBA001"
    row0 = df.row(0)
    game_id_idx = df.columns.index("game_id")
    assert row0[game_id_idx] == "NBA001"

def test_basketball_dataset_load(tmp_path):
    """
    Test BasketballDataset with JSON data represented as a dictionary.
    The data contains:
      - A game identifier ("test_game")
      - One event with one moment that includes two players.
    """
    data = {
        "gameid": "test_game",
        "events": [
            {
                "moments": [
                    [1, None, "12.0", "24", None, [
                        ["LAL", "Player1", 47, 25],
                        ["LAL", "Player2", 30, 10]
                    ]]
                ]
            }
        ]
    }
    file_path = tmp_path / "test_game.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    dataset = BasketballDataset(str(file_path))
    df = dataset.load()
    expected_columns = {"game_id", "event_id", "frame_id", "quarter", "game_clock", "shot_clock", "team", "player", "x", "y"}
    assert set(df.columns) >= expected_columns
    assert df.height == 2
    row0 = df.row(0)
    game_id_idx = df.columns.index("game_id")
    assert row0[game_id_idx] == "test_game"

def test_dataset_load_error(tmp_path):
    """
    Test BasketballDataset: ensure that loading a nonexistent file raises FileNotFoundError.
    """
    fake_file = tmp_path / "nonexistent.json"
    dataset = BasketballDataset(str(fake_file))
    with pytest.raises(FileNotFoundError):
        dataset.load()

def test_get_dataframe_without_load(sample_basketball_json):
    """
    Test BasketballDataset: calling get_dataframe() before load() should raise ValueError.
    """
    dataset = BasketballDataset(str(sample_basketball_json))
    with pytest.raises(ValueError):
        _ = dataset.get_dataframe()

def test_graph_settings_defaults():
    """
    Test BasketballGraphSettings: verify the default setting values.
    """
    settings = BasketballGraphSettings()
    settings_dict = settings.as_dict()
    assert settings_dict["self_loop_ball"] is True
    assert settings_dict["adjacency_matrix_connect_type"] == "ball"
    assert settings_dict["adjacency_matrix_type"] == "split_by_team"
    assert settings_dict["label_type"] == "binary"
    assert settings_dict["max_player_speed"] == 20.0
    assert settings_dict["max_ball_speed"] == 30.0
    assert settings_dict["normalize_coordinates"] is True
    assert settings_dict["defending_team_node_value"] == 0.0
    assert settings_dict["attacking_team_node_value"] == 1.0

def test_pitch_dimensions_defaults():
    """
    Test BasketballPitchDimensions: verify the official NBA court dimensions.
    """
    pdims = BasketballPitchDimensions()
    dims_dict = pdims.as_dict()
    assert dims_dict["court_length"] == 94.0
    assert dims_dict["court_width"] == 50.0
    assert dims_dict["three_point_radius"] == 23.75
    assert dims_dict["basket_x"] == 90.0
    assert dims_dict["basket_y"] == 25.0

def test_graph_converter_convert():
    """
    Test BasketballGraphConverter: create a synthetic dataset with 4 records (two rows for each frame_id)
    and verify that graph frames are generated correctly.
    """
    data = pl.DataFrame([
        {"frame_id": 1, "team": "Lakers", "x": 50.0, "y": 25.0},
        {"frame_id": 1, "team": "Lakers", "x": 52.0, "y": 26.0},
        {"frame_id": 2, "team": "Celtics", "x": 30.0, "y": 20.0},
        {"frame_id": 2, "team": "Celtics", "x": 32.0, "y": 21.0}
    ])

    class DummyDataset:
        pass
    dummy_dataset = DummyDataset()
    dummy_dataset.data = data

    settings = BasketballGraphSettings(self_loop_ball=False, normalize_coordinates=True)
    pdims = BasketballPitchDimensions()
    converter = BasketballGraphConverter(dummy_dataset, settings, pdims)
    graph_frames = converter.convert()

    # Expect 2 graph frames (one for each unique frame_id)
    assert len(graph_frames) == 2
    for gf in graph_frames:
        # Check that required keys exist
        for key in ["id", "x", "a", "e"]:
            assert key in gf
        # Check node features: normalized x and y, expected shape (n_nodes, 2)
        x = gf["x"]
        assert isinstance(x, np.ndarray)
        assert x.shape[1] == 2
        # If frame id is 1, check normalization: for x=50, y=25 with court_length=94 and court_width=50, expect approx (50/94, 25/50)
        if gf["id"] == 1:
            np.testing.assert_allclose(x[0], [50.0 / 94.0, 25.0 / 50.0], rtol=1e-2)
        # Check that the adjacency matrix is square and its size matches the number of nodes
        A = gf["a"]
        n_nodes = x.shape[0]
        assert A.shape[0] == A.shape[1] == n_nodes
        # Check edge features shape: should be (n_nodes * n_nodes, 1)
        e = gf["e"]
        assert e.shape[0] == n_nodes * n_nodes
        assert e.shape[1] == 1
        # Verify that the frame id is one of the expected values (1 or 2)
        assert gf["id"] in [1, 2]

def test_graph_converter_self_loop():
    """
    Test BasketballGraphConverter with self_loop_ball enabled.
    Verify that the diagonal of the adjacency matrix has non-zero values.
    """
    data = pl.DataFrame([
        {"frame_id": 1, "team": "Lakers", "x": 50.0, "y": 25.0},
        {"frame_id": 1, "team": "Lakers", "x": 52.0, "y": 26.0}
    ])

    class DummyDataset:
        pass
    dummy_dataset = DummyDataset()
    dummy_dataset.data = data

    settings = BasketballGraphSettings()  # Default: self_loop_ball=True
    pdims = BasketballPitchDimensions()
    converter = BasketballGraphConverter(dummy_dataset, settings, pdims)
    graph_frames = converter.convert()
    for gf in graph_frames:
        A = gf["a"].toarray()  # Convert csr_matrix to array for testing
        diag = np.diag(A)
        assert np.all(diag > 0)
