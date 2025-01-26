from dataclasses import dataclass, field

from typing import List

import polars as pl

import numpy as np

from .graph_settings import AmericanFootballPitchDimensions, Dimension, Unit
from ...utils import DefaultDataset, add_dummy_label_column, add_graph_id_column


class Constant:
    BALL = "football"
    QB = "QB"


class Column:
    OBJECT_ID = "nflId"

    GAME_ID = "gameId"
    FRAME_ID = "frameId"
    PLAY_ID = "playId"

    X = "x"
    Y = "y"

    ACCELERATION = "a"
    SPEED = "s"
    ORIENTATION = "o"
    DIRECTION = "dir"
    TEAM = "team"
    CLUB = "club"
    OFFICIAL_POSITION = "officialPosition"
    POSSESSION_TEAM = "possessionTeam"
    HEIGHT_CM = "height_cm"
    WEIGHT_KG = "weight_kg"


class Group:
    BY_FRAME = [Column.GAME_ID, Column.PLAY_ID, Column.FRAME_ID]
    BY_PLAY_POSSESSION_TEAM = [Column.GAME_ID, Column.PLAY_ID, Column.POSSESSION_TEAM]


@dataclass
class BigDataBowlDataset(DefaultDataset):
    def __init__(
        self,
        tracking_file_path: str,
        players_file_path: str,
        plays_file_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tracking_file_path = tracking_file_path
        self.players_file_path = players_file_path
        self.plays_file_path = plays_file_path
        self.pitch_dimensions = AmericanFootballPitchDimensions()

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
            df = df.with_columns(pl.col(Column.CLUB).alias(Column.TEAM))
            df = df.drop(Column.CLUB)

        df = (
            df.with_columns(
                pl.when(pl.col("playDirection") == play_direction)
                .then(pl.col(Column.ORIENTATION) + 180)  # rotate 180 degrees
                .otherwise(pl.col(Column.ORIENTATION))
                .alias(Column.ORIENTATION),
                pl.when(pl.col("playDirection") == play_direction)
                .then(pl.col(Column.DIRECTION) + 180)  # rotate 180 degrees
                .otherwise(pl.col(Column.DIRECTION))
                .alias(Column.DIRECTION),
            )
            .with_columns(
                [
                    (pl.col(Column.X) - (pitch_length / 2)).alias(Column.X),
                    (pl.col(Column.Y) - (pitch_width / 2)).alias(Column.Y),
                    # convert to radian on (-pi, pi) range
                    (
                        ((pl.col(Column.ORIENTATION) * np.pi / 180) + np.pi)
                        % (2 * np.pi)
                        - np.pi
                    ).alias(Column.ORIENTATION),
                    (
                        ((pl.col(Column.DIRECTION) * np.pi / 180) + np.pi) % (2 * np.pi)
                        - np.pi
                    ).alias(Column.DIRECTION),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("playDirection") == play_direction)
                    .then(pl.col(Column.X) * -1.0)
                    .otherwise(pl.col(Column.X))
                    .alias(Column.X),
                    pl.when(pl.col("playDirection") == play_direction)
                    .then(pl.col(Column.Y) * -1.0)
                    .otherwise(pl.col(Column.Y))
                    .alias(Column.Y),
                    # set "football" to nflId -9999 for ordering purposes
                    pl.when(pl.col(Column.TEAM) == Constant.BALL)
                    .then(-9999.9)
                    .otherwise(pl.col(Column.OBJECT_ID))
                    .alias(Column.OBJECT_ID),
                ]
            )
        )
        players = pl.read_csv(
            self.players_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            schema_overrides={"birthDate": pl.Date},
            ignore_errors=True,
        )
        if "position" in players.columns:
            players = players.with_columns(
                pl.col("position").alias(Column.OFFICIAL_POSITION)
            )
            players = players.drop("position")

        players = players.with_columns(
            pl.col(Column.OBJECT_ID)
            .cast(pl.Float64, strict=False)
            .alias(Column.OBJECT_ID)
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
            (
                players.select(
                    [
                        Column.OBJECT_ID,
                        Column.OFFICIAL_POSITION,
                        Column.HEIGHT_CM,
                        Column.WEIGHT_KG,
                    ]
                )
            ),
            on=Column.OBJECT_ID,
            how="left",
        )
        df = df.join(
            (plays.select(Group.BY_PLAY_POSSESSION_TEAM)),
            on=[Column.GAME_ID, Column.PLAY_ID],
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
        self, by: List[str] = ["gameId", "playId", "frameId"]
    ) -> pl.DataFrame:
        self.data = add_dummy_label_column(self.data, by, self._label_column)
        return self.data

    def add_graph_ids(self, by: List[str] = ["gameId", "playId"]) -> pl.DataFrame:
        self.data = add_graph_id_column(self.data, by, self._graph_id_column)
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
                (pl.col("feet") * 30.48 + pl.col("inches") * 2.54).alias(
                    Column.HEIGHT_CM
                ),
                (pl.col("weight") * 0.453592).alias(
                    Column.WEIGHT_KG
                ),  # Convert pounds to kilograms
            ]
        ).drop(["height", "feet", "inches", "weight"])
        return df
