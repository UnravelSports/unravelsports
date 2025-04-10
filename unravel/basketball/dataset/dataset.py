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
    Class for loading NBA tracking data.
    
    Modes:
      - URL: Loads from a 7zip archive (expects a JSON file inside).
      - Local: Loads from a file path or game identifier.
    """
    def __init__(self, source: str):
        self.source = source
        self.data = None

    def load(self) -> pl.DataFrame:
        """Loads and processes data into a Polars DataFrame with columns: game_id, frame_id, team, player, x, y."""
        if self.source.startswith("http"):
            if py7zr is None:
                raise ImportError("py7zr is required to extract 7zip archives.")
            response = requests.get(self.source)
            if response.status_code != 200:
                raise Exception("Failed to download data from URL.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".7z") as tmp_file:
                tmp_file.write(response.content)
                tmp_filename = tmp_file.name
            with py7zr.SevenZipFile(tmp_filename, mode='r') as archive:
                extract_path = tempfile.mkdtemp()
                archive.extractall(path=extract_path)
            os.unlink(tmp_filename)
            json_file = next((os.path.join(extract_path, fname) for fname in os.listdir(extract_path) if fname.endswith('.json')), None)
            if json_file is None:
                raise FileNotFoundError("JSON file not found in extracted archive.")
            with open(json_file, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)
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
        game_id = json_data.get("gameid", "unknown")
        events = json_data.get("events", [])
        for event in events:
            if "moments" in event:
                for m_idx, moment in enumerate(event["moments"]):
                    if len(moment) >= 6:
                        entities = moment[5]
                        for entity in entities[1:]:
                            if len(entity) >= 4:
                                rows.append({
                                    "game_id": game_id,
                                    "frame_id": m_idx,
                                    "team": entity[0],
                                    "player": entity[1],
                                    "x": entity[2],
                                    "y": entity[3]
                                })
            elif isinstance(json_data, list):
                for rec in json_data:
                    rows.append({
                        "game_id": rec.get("game_id", game_id),
                        "frame_id": rec.get("frame_id"),
                        "team": rec.get("team"),
                        "player": rec.get("player"),
                        "x": rec.get("x"),
                        "y": rec.get("y")
                    })
        self.data = pl.DataFrame(rows)
        return self.data

    def get_dataframe(self) -> pl.DataFrame:
        """Returns the loaded DataFrame; load() must be called first."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data
