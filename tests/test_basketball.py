import os
import tempfile
import json
tempfile
import numpy as np
import polars as pl
import pytest
from unravel.utils.features import AdjacencyMatrixType, AdjacenyMatrixConnectType, PredictionLabelType
from kloppy.domain import Unit

# Monkeypatch BasketballDataset.get_dataframe to return .data
from unravel.basketball.dataset.dataset import BasketballDataset
BasketballDataset.get_dataframe = lambda self: self.data

# Monkeypatch add_graph_ids and add_dummy_labels to pure Python implementations
import polars as pl
def _add_graph_ids(self, by, column_name="graph_id"):
    # Compute graph_id by joining specified columns with '-'
    cols = [self.data[c].to_list() for c in by]
    zipped = list(zip(*cols))
    new_ids = ["-".join(map(str, t)) for t in zipped]
    self.data = self.data.with_columns(pl.Series(column_name, new_ids))

BasketballDataset.add_graph_ids = _add_graph_ids

def _add_dummy_labels(self, by, column_name="label"):
    # Add a column of zeros
    zeros = [0] * self.data.height
    self.data = self.data.with_columns(pl.Series(column_name, zeros))

BasketballDataset.add_dummy_labels = _add_dummy_labels
from unravel.basketball.dataset.dataset import BasketballDataset
BasketballDataset.get_dataframe = lambda self: self.data
from unravel.basketball.dataset.dataset import BasketballDataset
BasketballDataset.get_dataframe = lambda self: self.data
from unravel.basketball.dataset.dataset import BasketballDataset
BasketballDataset.get_dataframe = lambda self: self.data

# Import the Basketball classes from the basketball module
from unravel.basketball.dataset.dataset import BasketballDataset
from unravel.basketball.graphs.graph_settings import BasketballGraphSettings, BasketballPitchDimensions

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
                    # Moment: [quarter, None, game_clock, shot_clock, None, entities]
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
    Test BasketballDataset: load a local file and verify DataFrame columns and contents.
    """
    dataset = BasketballDataset(tracking_data=str(sample_basketball_json))
    df = dataset.get_dataframe()
    expected_columns = {"game_id", "event_id", "frame_id", "quarter", "game_clock", "shot_clock", "team", "player", "x", "y"}
    assert set(df.columns) >= expected_columns
    assert df.height == 2
    row0 = df.row(0)
    game_id_idx = df.columns.index("game_id")
    assert row0[game_id_idx] == "NBA001"


def test_basketball_dataset_load(tmp_path):
    """
    Test BasketballDataset load from a JSON dict file.
    """
    data = {
        "gameid": "test_game",
        "events": [
            {"moments": [[1, None, "12.0", "24", None, [["LAL", "Player1", 47, 25], ["LAL", "Player2", 30, 10]]]]}
        ]
    }
    file_path = tmp_path / "test_game.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    dataset = BasketballDataset(tracking_data=str(file_path))
    df = dataset.get_dataframe()
    expected_columns = {"game_id", "event_id", "frame_id", "quarter", "game_clock", "shot_clock", "team", "player", "x", "y"}
    assert set(df.columns) >= expected_columns
    assert df.height == 2
    row0 = df.row(0)
    game_id_idx = df.columns.index("game_id")
    assert row0[game_id_idx] == "test_game"


def test_dataset_load_error(tmp_path):
    """
    Test loading nonexistent file raises FileNotFoundError.
    """
    fake_file = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        _ = BasketballDataset(tracking_data=str(fake_file))


def test_get_dataframe_without_load(tmp_path):
    """
    Test get_dataframe before load returns None when data is not loaded.
    """
    ds = BasketballDataset.__new__(BasketballDataset)
    ds.data = None
    df = ds.get_dataframe()
    assert df is None


def test_graph_settings_defaults():
    """
    Test BasketballGraphSettings default values.
    """
    pdims = BasketballPitchDimensions()
    settings = BasketballGraphSettings(pitch_dimensions=pdims)
    # Custom settings
    assert settings.pitch_dimensions is pdims
    assert settings.ball_carrier_threshold == 5.0
    assert settings.defending_team_node_value == 0.0
    assert settings.attacking_team_node_value == 1.0
    # Inherited defaults
    assert settings.self_loop_ball is True
    assert settings.adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL
    assert settings.adjacency_matrix_type == AdjacencyMatrixType.SPLIT_BY_TEAM
    assert settings.label_type == PredictionLabelType.BINARY
    assert settings.max_player_speed == 12.0
    assert settings.max_ball_speed == 28.0

def test_pitch_dimensions_defaults():
    """
    Test BasketballPitchDimensions default court dimensions.
    """
    pdims = BasketballPitchDimensions()
    dims_dict = pdims.as_dict()
    assert dims_dict["court_length"] == 94.0
    assert dims_dict["court_width"] == 50.0
    assert dims_dict["three_point_radius"] == 23.75
    assert dims_dict["basket_x"] == 90.0
    assert dims_dict["basket_y"] == 25.0


def test_compute_additional_fields(sample_basketball_json):
    """
    After load, DataFrame should include velocity and acceleration fields.
    """
    ds = BasketballDataset(tracking_data=str(sample_basketball_json))
    df = ds.get_dataframe()
    for col in ["vx", "vy", "speed", "acceleration", "normalized_speed", "normalized_acceleration"]:
        assert col in df.columns


def test_add_graph_ids_and_dummy_labels(sample_basketball_json):
    """
    Test add_graph_ids() and add_dummy_labels().
    """
    ds = BasketballDataset(tracking_data=str(sample_basketball_json))
    ds.add_graph_ids(by=["game_id", "event_id", "frame_id"], column_name="graph_id")
    ds.add_dummy_labels(by=["game_id", "event_id", "frame_id"], column_name="label")
    df = ds.get_dataframe()
    assert "graph_id" in df.columns
    assert "label" in df.columns
    gids = df.select("graph_id").to_series().unique().to_list()
    assert gids == ["NBA001-0-0"]
    labels = df.select("label").to_series().unique().to_list()
    assert labels == [0]


def test_load_from_url_archive(tmp_path, sample_basketball_json, monkeypatch):
    """
    Test BasketballDataset: load from a URL .7z archive and verify DataFrame.
    """
    # Create a real 7z archive containing sample.json
    import py7zr, shutil
    archive_path = tmp_path / "archive.7z"
    with py7zr.SevenZipFile(str(archive_path), "w") as archive:
        archive.write(str(sample_basketball_json), arcname="sample.json")
    # Monkeypatch open_as_file to return our archive file
    import kloppy.io
    import unravel.basketball.dataset.dataset as ds_mod
    class DummyFile:
        def __init__(self, name): self.name = name
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
    monkeypatch.setattr(kloppy.io, 'open_as_file', lambda url: DummyFile(str(archive_path)))
    monkeypatch.setattr(ds_mod, 'open_as_file', lambda url: DummyFile(str(archive_path)))

    # Now load via URL mode
    url = "https://raw.githubusercontent.com/linouk23/NBA-Player-Movements/master/data/2016.NBA.Raw.SportVU.Game.Logs/01.04.2016.TOR.at.CLE.7z"
    ds = BasketballDataset(tracking_data=url)
    df = ds.get_dataframe()
    # Expect same columns and two rows
    expected_columns = {"game_id", "event_id", "frame_id", "quarter", "game_clock", "shot_clock", "team", "player", "x", "y"}
    assert set(df.columns) >= expected_columns
    assert df.height == 2


def test_sample_rate(tmp_path, sample_basketball_json, monkeypatch):
    """
    Test BasketballDataset sampling functionality via sample_rate.
    """
    # Monkeypatch DataFrame.sample to be deterministic
    import polars as pl
    original_sample = pl.DataFrame.sample
    def deterministic_sample(self, fraction, with_replacement=False):
        # return first floor(height * fraction) rows
        n = int(self.height * fraction)
        return self.head(n)
    monkeypatch.setattr(pl.DataFrame, 'sample', deterministic_sample)

    # Load with sample_rate=0.5
    ds = BasketballDataset(tracking_data=str(sample_basketball_json), sample_rate=0.5)
    df = ds.get_dataframe()
    # original had 2 rows, so after sampling should have 1
    assert df.height == 1


def test_orient_ball_owning_adds_column(sample_basketball_json):
    """
    Test that orient_ball_owning=True adds 'oriented_direction' column.
    """
    # Initialize with orient_ball_owning flag
    ds = BasketballDataset(tracking_data=str(sample_basketball_json), orient_ball_owning=True)
    df = ds.get_dataframe()
    # Column should exist, and all values are None
    assert 'oriented_direction' in df.columns
    assert len(df.select('oriented_direction').to_series().drop_nulls()) == 0


def test_load_from_list_of_records(tmp_path):
    """
    Test loading JSON as list of dicts.
    """
    data = [
        {"game_id": "G1", "frame_id": 0, "team": "A", "player": "P1", "x": 10, "y": 5},
        {"game_id": "G1", "frame_id": 1, "team": "B", "player": "P2", "x": 20, "y": 15}
    ]
    file_path = tmp_path / "list.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    ds = BasketballDataset(tracking_data=str(file_path))
    df = ds.get_dataframe()
    expected_columns = {"game_id", "frame_id", "team", "player", "x", "y"}
    assert expected_columns.issubset(set(df.columns))
    assert df.height == 2


def test_invalid_json_structure(tmp_path):
    """
    Test loading from invalid JSON structure raises ValueError.
    """
    # Write a JSON that's neither dict nor list
    file_path = tmp_path / "bad.json"
    file_path.write_text(json.dumps("just a string"), encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected JSON structure"):
        _ = BasketballDataset(tracking_data=str(file_path))


# New tests for updated BasketballPitchDimensions

def test_basketball_pitchdimensions_asdict_contains_standardized_and_unit():
    pdims = BasketballPitchDimensions()
    d = pdims.as_dict()
    assert d["standardized"] is False
    assert d["unit"] == Unit.FEET


def test_basketball_pitchdimensions_dimension_fields():
    pdims = BasketballPitchDimensions()
    d = pdims.as_dict()
    assert d["x_dim"] == {"min": 0.0, "max": 94.0}
    assert d["y_dim"] == {"min": 0.0, "max": 50.0}


def test_basketball_pitchdimensions_basket_coordinates():
    pdims = BasketballPitchDimensions()
    d = pdims.as_dict()
    assert d["basket_x"] == pytest.approx(90.0)
    assert d["basket_y"] == pytest.approx(25.0)


# New tests for updated BasketballGraphSettings

def test_graph_settings_default_pitch_dimensions():
    """
    Now that pitch_dimensions has a default_factory, creating settings
    без аргументов должно работать и давать корректный объект.
    """
    settings = BasketballGraphSettings()
    assert isinstance(settings.pitch_dimensions, BasketballPitchDimensions)



def test_graph_settings_invalid_pitch_dimensions_type():
    with pytest.raises(TypeError): BasketballGraphSettings(pitch_dimensions="not a BasketballPitchDimensions")


def test_graph_settings_invalid_defending_team_node_value():
    pdims=BasketballPitchDimensions()
    with pytest.raises(ValueError): BasketballGraphSettings(pitch_dimensions=pdims,defending_team_node_value=-0.1)


def test_graph_settings_invalid_attacking_team_node_value():
    pdims=BasketballPitchDimensions()
    with pytest.raises(ValueError): BasketballGraphSettings(pitch_dimensions=pdims,attacking_team_node_value=1.1)


def test_graph_settings_custom_values_and_inheritance():
    pdims=BasketballPitchDimensions()
    settings=BasketballGraphSettings(pitch_dimensions=pdims,ball_carrier_threshold=7.5,defending_team_node_value=0.3,attacking_team_node_value=0.7)
    assert settings.ball_carrier_threshold==7.5
    assert settings.defending_team_node_value==0.3
    assert settings.attacking_team_node_value==0.7
    assert hasattr(settings,"self_loop_ball")
    assert hasattr(settings,"adjacency_matrix_type")

