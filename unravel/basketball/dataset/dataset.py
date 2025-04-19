from dataclasses import dataclass, field
from typing import Optional
import os
import json
import tempfile
import polars as pl
import requests
import numpy as np

from kloppy.io import open_as_file

try:
    import py7zr
except ImportError:
    py7zr = None

from ...utils import DefaultDataset

@dataclass(kw_only=True)
class BasketballDataset(DefaultDataset):
    """
    Loads NBA tracking data.
    
    Modes:
      - URL: Loads from a 7zip archive (expects a JSON file inside).
      - Local: Loads from a file path or game identifier.
      
    Additional parameters:
      - max_player_speed, max_ball_speed, max_player_acceleration, max_ball_acceleration:
          Threshold values for normalizing player and ball speeds/accelerations.
      - orient_ball_owning:
          Flag indicating whether to compute oriented direction for ball ownership.
      - sample_rate:
          Fraction of data to sample (e.g., 0.5 to keep half of the rows).

    """
    tracking_data: str
    max_player_speed: float = 20.0
    max_ball_speed: float = 30.0
    max_player_acceleration: float = 10.0
    max_ball_acceleration: float = 10.0
    orient_ball_owning: bool = False
    sample_rate: float = 1.0
    data: Optional[pl.DataFrame] = field(default=None, init=False)

    def load(self) -> pl.DataFrame:
        """
        Loads JSON data from a local file or URL and converts it into a Polars DataFrame with the following columns:
          game_id, event_id, frame_id, quarter, game_clock, shot_clock, team, player, x, y.
        """
        # Load via URL if tracking_data starts with "http"
        if self.tracking_data.startswith("http"):
            with open_as_file(self.tracking_data) as tmp_file:
                tmp_filename = tmp_file.name
            json_file = None
            if py7zr is None:
                raise ImportError("py7zr is required to extract 7zip archives.")
            with py7zr.SevenZipFile(tmp_filename, mode='r') as archive:
                for fname in archive.getnames():
                    if fname.endswith('.json'):
                        json_file = os.path.join(tempfile.mkdtemp(), fname)
                        with open(json_file, 'wb') as f:
                            file_dict = archive.read(fname)
                            file_bytes = file_dict[fname].read()
                            f.write(file_bytes)
                        break
            os.unlink(tmp_filename)
            if json_file is None:
                raise FileNotFoundError("JSON file not found in extracted archive.")
            with open(json_file, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
        else:
            # Load from file if a valid file path is provided
            if os.path.isfile(self.tracking_data):
                file_path = self.tracking_data
            else:
                # Search for a file in the default directory using the game identifier
                file_path = os.path.join("data", "nba", f"{self.tracking_data}.json")
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Game file '{self.tracking_data}.json' not found at: {file_path}")
                
            with open_as_file(file_path) as f:
                json_data = json.load(f)

        rows = []
        # Process JSON as a dictionary
        if isinstance(json_data, dict):
            game_id = json_data.get("gameid", "unknown")
            events = json_data.get("events", [])
            for event_id, event in enumerate(events):
                if "moments" in event:
                    for m_idx, moment in enumerate(event["moments"]):
                        if len(moment) >= 6:
                            quarter = moment[0]
                            game_clock = moment[2]
                            shot_clock = moment[3]
                            for entity in moment[5]:
                                if len(entity) >= 4:
                                    rows.append({
                                        "game_id": game_id,
                                        "event_id": event_id,
                                        "frame_id": m_idx,
                                        "quarter": quarter,
                                        "game_clock": float(game_clock) if game_clock is not None else None,
                                        "shot_clock": float(shot_clock) if shot_clock is not None else None,
                                        "team": entity[0],
                                        "player": entity[1],
                                        "x": float(entity[2]),
                                        "y": float(entity[3])
                                    })
        # Process JSON if it is a list
        elif isinstance(json_data, list):
            for rec in json_data:
                rows.append({
                    "game_id": rec.get("game_id", "unknown"),
                    "frame_id": rec.get("frame_id"),
                    "team": rec.get("team"),
                    "player": rec.get("player"),
                    "x": float(rec.get("x", 0)),
                    "y": float(rec.get("y", 0))
                })
        else:
            raise ValueError("Unexpected JSON structure")

        self.data = pl.DataFrame(rows, strict=False)
        if self.sample_rate < 1.0:
            self.data = self.data.sample(fraction=self.sample_rate, with_replacement=False)

        return self.data

    def compute_additional_fields(self) -> pl.DataFrame:
        """
        After loading the data (via load()), compute additional fields:
          - vx, vy: velocity components,
          - speed: magnitude of the velocity,
          - direction: movement direction in radians,
          - acceleration: change in speed over time,
          - normalized_speed: speed normalized by max_player_speed or max_ball_speed,
          - normalized_acceleration: acceleration normalized by max_player_acceleration or max_ball_acceleration,
          - oriented_direction (if orient_ball_owning is True): a placeholder for ball-owning orientation.
        
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.data.sort(["game_id", "player", "frame_id"])
        if "game_clock" in df.columns:
            df = df.with_columns([
                (pl.col("game_clock").shift(-1) - pl.col("game_clock")).abs().alias("dt_temp")
            ])
            df = df.with_columns([
                pl.col("dt_temp").fill_null(1).alias("dt")
            ])
        else:
            df = df.with_columns([pl.lit(1).alias("dt")])
        
        df = df.with_columns([
            (pl.col("x") - pl.col("x").shift(1)).alias("dx"),
            (pl.col("y") - pl.col("y").shift(1)).alias("dy")
        ])
        
        df = df.with_columns([
            (pl.col("dx") / pl.col("dt")).alias("vx"),
            (pl.col("dy") / pl.col("dt")).alias("vy")
        ])

        df = df.with_columns([
            ((pl.col("vx") ** 2 + pl.col("vy") ** 2) ** 0.5).alias("speed")
        ])

        df = df.with_columns(
                            pl.concat_list([pl.col("vx"), pl.col("vy")])
                            .map_elements(lambda row: float(np.arctan2(row[1], row[0])) if row[0] is not None and row[1] is not None else None)
                            .alias("direction")
                        )

        if "game_clock" in df.columns:
            df = df.with_columns([
                (pl.col("game_clock") - pl.col("game_clock").shift(1)).abs().alias("dt_acc")
            ])
            df = df.with_columns([
                ((pl.col("speed") - pl.col("speed").shift(1)) / pl.col("dt_acc")).alias("acceleration")
            ])
        else:
            df = df.with_columns([
                ((pl.col("speed") - pl.col("speed").shift(1)) / 1).alias("acceleration")
            ])
        return df

    def get_dataframe(self) -> pl.DataFrame:
        """Returns the loaded DataFrame; ensure load() is called first."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data
