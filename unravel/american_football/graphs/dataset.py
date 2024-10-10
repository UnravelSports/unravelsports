from dataclasses import dataclass, field

from typing import List

import polars as pl

import numpy as np

from .graph_settings import AmericanFootballPitchDimensions
from ...utils import add_dummy_label_column, add_graph_id_column


@dataclass
class BigDataBowlDataset:
    tracking_file_path: str
    players_file_path: str
    plays_file_path: str
    pitch_dimensions: AmericanFootballPitchDimensions = field(
        init=False, repr=False, default_factory=AmericanFootballPitchDimensions
    )

    def __post_init__(self):
        if (
            not self.tracking_file_path
            or not self.players_file_path
            or not self.plays_file_path
        ):
            raise Exception("Missing data file path...")

    def load(self):
        df = pl.read_csv(
            self.tracking_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            try_parse_dates=True,
        )
        df = df.with_columns(
            [
                (pl.col("x") - (self.pitch_dimensions.pitch_length / 2)).alias("x"),
                (pl.col("y") - (self.pitch_dimensions.pitch_width / 2)).alias("y"),
                ((pl.col("o") * np.pi / 180 + np.pi) % (2 * np.pi) - np.pi).alias("o"),
                ((pl.col("dir") * np.pi / 180 + np.pi) % (2 * np.pi) - np.pi).alias(
                    "dir"
                ),
            ]
        ).with_columns(
            [
                pl.when(pl.col("playDirection") == "left")
                .then(pl.col("x") * -1.0)
                .otherwise(pl.col("x"))
                .alias("x"),
                pl.when(pl.col("playDirection") == "left")
                .then(pl.col("y") * -1.0)
                .otherwise(pl.col("y"))
                .alias("y"),
                pl.when(pl.col("playDirection") == "left")
                .then(pl.col("o") * -1.0)
                .otherwise(pl.col("o"))
                .alias("o"),
                pl.when(pl.col("playDirection") == "left")
                .then(pl.col("dir") * -1.0)
                .otherwise(pl.col("dir"))
                .alias("dir"),
                # set "football" to nflId -9999 for ordering purposes
                pl.when(pl.col("team") == "football")
                .then(-9999)
                .otherwise(pl.col("nflId"))
                .alias("nflId"),
            ]
        )

        players = pl.read_csv(
            self.players_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            try_parse_dates=True,
        )

        plays = pl.read_csv(
            self.plays_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            try_parse_dates=True,
        )

        df = df.join(
            (players.select(["nflId", "officialPosition"])), on="nflId", how="left"
        )
        df = df.join(
            (plays.select(["gameId", "playId", "possessionTeam"])),
            on=["gameId", "playId"],
            how="left",
        )
        self.data = df
        return self.data, self.pitch_dimensions

    def add_dummy_labels(
        self,
        by: List[str] = ["gameId", "playId", "frameId"],
        column_name: str = "label",
    ) -> pl.DataFrame:
        self.data = add_dummy_label_column(self.data, by, column_name)
        return self.data

    def add_graph_ids(
        self, by: List[str] = ["gameId", "playId"], column_name: str = "graph_id"
    ) -> pl.DataFrame:
        self.data = add_graph_id_column(self.data, by, column_name)
        return self.data
