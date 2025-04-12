from dataclasses import dataclass, field
from typing import Optional
import os
import json
import tempfile
import polars as pl
import requests
import numpy as np

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
    """
    tracking_data: str
    data: Optional[pl.DataFrame] = field(default=None, init=False)

    def load(self) -> pl.DataFrame:
        """
        Loads JSON data from a local file or URL and converts it into a Polars DataFrame with the following columns:
          game_id, event_id, frame_id, quarter, game_clock, shot_clock, team, player, x, y.
        """
        # Load via URL if tracking_data starts with "http"
        if self.tracking_data.startswith("http"):
            if py7zr is None:
                raise ImportError("py7zr is required to extract 7zip archives.")
            response = requests.get(self.tracking_data)
            if response.status_code != 200:
                raise Exception("Failed to download data from URL.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".7z") as tmp_file:
                tmp_file.write(response.content)
                tmp_filename = tmp_file.name
            with py7zr.SevenZipFile(tmp_filename, mode='r') as archive:
                extract_path = tempfile.mkdtemp()
                archive.extractall(path=extract_path)
            os.unlink(tmp_filename)
            json_file = next(
                (os.path.join(extract_path, fname) for fname in os.listdir(extract_path) if fname.endswith('.json')),
                None
            )
            if json_file is None:
                raise FileNotFoundError("JSON file not found in extracted archive.")
            with open(json_file, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
        else:
            # Load from file if a valid file path is provided
            if os.path.isfile(self.tracking_data):
                with open(self.tracking_data, 'r', encoding='utf-8') as jf:
                    json_data = json.load(jf)
            else:
                file_path = os.path.join("data", "nba", f"{self.source}.json")
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Game file '{self.tracking_data}.json' not found at: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as jf:
                    json_data = json.load(jf)

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
        return self.data

    def compute_additional_fields(self) -> pl.DataFrame:
        """
        After loading the data (via load()), compute additional fields:
          - vx, vy: velocity components,
          - speed: magnitude of the velocity,
          - direction: movement direction in radians,
          - acceleration: change in speed over time.
        
        Calculations are performed for each group defined by game_id and player.
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
        df = df.with_columns([
            pl.struct(["vx", "vy"]).apply(
                lambda row: float(np.arctan2(row["vy"], row["vx"]))
                if (row["vx"] is not None and row["vy"] is not None) else None
            ).alias("direction")
        ])
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
