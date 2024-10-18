from dataclasses import dataclass, field

from typing import List

import polars as pl

import numpy as np

from .graph_settings import AmericanFootballPitchDimensions, Dimension, Unit
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
        pitch_length = self.pitch_dimensions.pitch_length
        pitch_width = self.pitch_dimensions.pitch_width

        df = pl.read_csv(
            self.tracking_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            try_parse_dates=True,
        )

        play_direction = "left"

        if "club" in df.columns:
            df = df.with_columns(pl.col("club").alias("team"))
            df = df.drop("club")

        df = (
            df.with_columns(
                pl.when(pl.col("playDirection") == play_direction)
                .then(pl.col("o") + 180)  # rotate 180 degrees
                .otherwise(pl.col("o"))
                .alias("o"),
                pl.when(pl.col("playDirection") == play_direction)
                .then(pl.col("dir") + 180)  # rotate 180 degrees
                .otherwise(pl.col("dir"))
                .alias("dir"),
            )
            .with_columns(
                [
                    (pl.col("x") - (pitch_length / 2)).alias("x"),
                    (pl.col("y") - (pitch_width / 2)).alias("y"),
                    # convert to radian on (-pi, pi) range
                    (((pl.col("o") * np.pi / 180) + np.pi) % (2 * np.pi) - np.pi).alias(
                        "o"
                    ),
                    (
                        ((pl.col("dir") * np.pi / 180) + np.pi) % (2 * np.pi) - np.pi
                    ).alias("dir"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("playDirection") == play_direction)
                    .then(pl.col("x") * -1.0)
                    .otherwise(pl.col("x"))
                    .alias("x"),
                    pl.when(pl.col("playDirection") == play_direction)
                    .then(pl.col("y") * -1.0)
                    .otherwise(pl.col("y"))
                    .alias("y"),
                    # set "football" to nflId -9999 for ordering purposes
                    pl.when(pl.col("team") == "football")
                    .then(-9999.9)
                    .otherwise(pl.col("nflId"))
                    .alias("nflId"),
                ]
            )
        )
        players = pl.read_csv(
            self.players_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            dtypes={"birthDate": pl.Date},
            ignore_errors=True,
        )
        if "position" in players.columns:
            players = players.with_columns(pl.col("position").alias("officialPosition"))
            players = players.drop("position")

        players = players.with_columns(
            pl.col("nflId").cast(pl.Float64, strict=False).alias("nflId")
        )
        players = self._convert_weight_height_to_metric(df=players)

        plays = pl.read_csv(
            self.plays_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            try_parse_dates=True,
        )

        df = df.join(
            (players.select(["nflId", "officialPosition", "height_cm", "weight_kg"])),
            on="nflId",
            how="left",
        )
        df = df.join(
            (plays.select(["gameId", "playId", "possessionTeam"])),
            on=["gameId", "playId"],
            how="left",
        )
        self.data = df

        # update pitch dimensions to how it looks after loading
        self.pitch_dimensions = AmericanFootballPitchDimensions(
            x_dim=Dimension(min=-pitch_length / 2, max=pitch_length / 2),
            y_dim=Dimension(min=-pitch_width / 2, max=pitch_width / 2),
            standardized=False,
            unit=Unit.YARDS,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

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

    @staticmethod
    def _convert_weight_height_to_metric(df: pl.DataFrame):
        df = df.with_columns(
            [
                pl.col("height")
                .str.extract(r"(\d+)")
                .cast(pl.Float64)
                .alias("feet"),  # Extract feet and cast to float
                pl.col("height")
                .str.extract(r"\d+-(\d+)", 1)
                .cast(pl.Float64)
                .alias("inches"),  # Extract inches and cast to float
            ]
        )
        df = df.with_columns(
            [
                (pl.col("feet") * 30.48 + pl.col("inches") * 2.54).alias("height_cm"),
                (pl.col("weight") * 0.453592).alias(
                    "weight_kg"
                ),  # Convert pounds to kilograms
            ]
        ).drop(["height", "feet", "inches", "weight"])
        return df
