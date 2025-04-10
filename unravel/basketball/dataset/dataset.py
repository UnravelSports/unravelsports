import os
import json
import tempfile
import polars as pl
import requests

try:
    import py7zr
except ImportError:
    py7zr = None

class BasketballDataset:
    """
    Loads NBA tracking data.
    
    Modes:
      - URL: Loads from a 7zip archive (expects a JSON file inside).
      - Local: Loads from a file path or game identifier.
    """
    def __init__(self, source: str):
        self.source = source
        self.data = None

    def load(self) -> pl.DataFrame:
        # Загрузка JSON из файла или по URL остаётся без изменений…
        if self.source.startswith("http"):
            # [код для загрузки по URL]
            ...
        else:
            if os.path.isfile(self.source):
                with open(self.source, 'r', encoding='utf-8') as jf:
                    json_data = json.load(jf)
            else:
                file_path = os.path.join("data", "nba", f"{self.source}.json")
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Game file '{self.source}.json' not found at: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as jf:
                    json_data = json.load(jf)
        
        rows = []
        # Если json_data - словарь, обрабатываем как ранее:
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
        # Если же json_data - список, обрабатываем каждую запись отдельно:
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
          
        The method groups the data by game_id and player, sorting by frame_id.
        If the column 'game_clock' exists, the time difference is computed based on it,
        keeping in mind that the game clock is usually counting down.
        Otherwise, a default dt = 1 is used.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        # Sort the data by game_id, player, and frame_id to ensure correct temporal order.
        df = self.data.sort(["game_id", "player", "frame_id"])

        # Compute time difference for velocity calculation.
        # For velocity, we use the time difference from the current row to the next row.
        if "game_clock" in df.columns:
            # Calculate dt_temp as the absolute difference between the current and next game_clock.
            df = df.with_columns([
                (pl.col("game_clock").shift(-1) - pl.col("game_clock")).abs().alias("dt_temp")
            ])
            # Use dt = 1 if dt_temp is null.
            df = df.with_columns([
                pl.col("dt_temp").fill_null(1).alias("dt")
            ])
        else:
            # Default dt value when no game_clock is available.
            df = df.with_columns([
                pl.lit(1).alias("dt")
            ])

        # Compute differences in x and y coordinates: current value minus previous value.
        df = df.with_columns([
            (pl.col("x") - pl.col("x").shift(1)).alias("dx"),
            (pl.col("y") - pl.col("y").shift(1)).alias("dy")
        ])

        # Calculate velocity components (vx, vy) by dividing the displacement by dt.
        df = df.with_columns([
            (pl.col("dx") / pl.col("dt")).alias("vx"),
            (pl.col("dy") / pl.col("dt")).alias("vy")
        ])

        # Compute speed as the Euclidean norm of (vx, vy).
        df = df.with_columns([
            ((pl.col("vx") ** 2 + pl.col("vy") ** 2) ** 0.5).alias("speed")
        ])

        # Calculate movement direction using arctan2(vy, vx).
        df = df.with_columns([
            pl.struct(["vx", "vy"]).apply(
                lambda row: float(np.arctan2(row["vy"], row["vx"])) 
                if (row["vx"] is not None and row["vy"] is not None) else None
            ).alias("direction")
        ])

        # Compute acceleration based on the change in speed over the time difference.
        # For acceleration, we calculate dt_acc as the absolute difference between the current and previous game_clock.
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
