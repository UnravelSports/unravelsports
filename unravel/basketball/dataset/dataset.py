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
          Thresholds for normalizing speed/acceleration.
      - orient_ball_owning:
          If True, computes oriented direction for ball ownership (placeholder).
      - sample_rate:
          Fraction of rows to sample (0.0–1.0).
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
        # Load JSON from URL or local
        if self.tracking_data.startswith("http"):
            with open_as_file(self.tracking_data) as tmp_file:
                tmp_filename = tmp_file.name
            if py7zr is None:
                raise ImportError("py7zr is required to extract 7zip archives.")
            json_file = None
            with py7zr.SevenZipFile(tmp_filename, mode='r') as archive:
                for fname in archive.getnames():
                    if fname.endswith('.json'):
                        extract_path = tempfile.mkdtemp()
                        archive.extract(targets=[fname], path=extract_path)
                        json_file = os.path.join(extract_path, fname)
                        break
            os.unlink(tmp_filename)
            if json_file is None:
                raise FileNotFoundError("JSON file not found in archive.")
            with open(json_file, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
        else:
            if os.path.isfile(self.tracking_data):
                file_path = self.tracking_data
            else:
                file_path = os.path.join("data", "nba", f"{self.tracking_data}.json")
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Game file '{self.tracking_data}.json' not found at {file_path}")
            with open_as_file(file_path) as f:
                json_data = json.load(f)

        rows = []
        if isinstance(json_data, dict):
            game_id = json_data.get("gameid", "unknown")
            for event_id, event in enumerate(json_data.get("events", [])):
                for m_idx, moment in enumerate(event.get("moments", [])):
                    if len(moment) >= 6:
                        quarter, _, game_clock, shot_clock, *_ , entities = moment
                        for entity in entities:
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

        # Build DataFrame and sample if needed
        self.data = pl.DataFrame(rows, strict=False)
        if self.sample_rate < 1.0:
            self.data = self.data.sample(fraction=self.sample_rate, with_replacement=False)

        # Compute velocities, speed, direction, acceleration, and normalized fields
        self.data = self.compute_additional_fields()
        return self.data

    def compute_additional_fields(self) -> pl.DataFrame:
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        df = self.data.sort(["game_id", "player", "frame_id"])

        # Time delta for velocity
        if "game_clock" in df.columns:
            df = df.with_columns((pl.col("game_clock").shift(-1) - pl.col("game_clock")).abs().fill_null(1).alias("dt"))
        else:
            df = df.with_columns(pl.lit(1).alias("dt"))

        # Displacements
        df = df.with_columns([
            (pl.col("x") - pl.col("x").shift(1)).alias("dx"),
            (pl.col("y") - pl.col("y").shift(1)).alias("dy")
        ])

        # Velocity components
        df = df.with_columns([
            (pl.col("dx") / pl.col("dt")).alias("vx"),
            (pl.col("dy") / pl.col("dt")).alias("vy")
        ])

        # Speed and acceleration
        df = df.with_columns([
            ((pl.col("vx")**2 + pl.col("vy")**2)**0.5).alias("speed")
        ])
        if "game_clock" in df.columns:
            df = df.with_columns((pl.col("game_clock") - pl.col("game_clock").shift(1)).abs().fill_null(1).alias("dt_acc"))
        else:
            df = df.with_columns(pl.lit(1).alias("dt_acc"))
        df = df.with_columns(((pl.col("speed") - pl.col("speed").shift(1)) / pl.col("dt_acc")).alias("acceleration"))

        # Normalize speed and acceleration
        df = df.with_columns([
            pl.when(pl.col("player").str.contains("ball", literal=False))
                .then(pl.col("speed") / self.max_ball_speed)
                .otherwise(pl.col("speed") / self.max_player_speed)
                .clip(0, 1)
                .alias("normalized_speed"),
            pl.when(pl.col("player").str.contains("ball", literal=False))
                .then(pl.col("acceleration") / self.max_ball_acceleration)
                .otherwise(pl.col("acceleration") / self.max_player_acceleration)
                .alias("normalized_acceleration")
        ])

        # Placeholder for oriented direction if requested
        if self.orient_ball_owning:
            df = df.with_columns(pl.lit(None).alias("oriented_direction"))

        return df

    
