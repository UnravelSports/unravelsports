from dataclasses import dataclass, field

from typing import List

import polars as pl

import numpy as np

from kloppy.domain import Dimension, Unit, Orientation

from ...utils import (
    DefaultSettings,
    DefaultDataset,
    AmericanFootballPitchDimensions,
    add_dummy_label_column,
    add_graph_id_column,
)

from .objects import Column, Group, Constant


@dataclass
class BigDataBowlDataset(DefaultDataset):
    """Load and preprocess NFL Big Data Bowl tracking data into Polars DataFrame format.

    This class handles NFL tracking data from the Big Data Bowl competition, converting
    CSV files into a standardized Polars DataFrame with computed velocities, standardized
    coordinate systems, and orientation normalization. It processes three input files:
    tracking data, player metadata, and play information.

    The loader performs:
    - Coordinate system standardization (centering at midfield)
    - Orientation normalization (attacking left-to-right)
    - Angle conversion (degrees → radians in [-π, π] range)
    - Player metadata enrichment (height, weight, position)
    - Play-level information joining (possession team, play details)
    - Metric conversion (imperial → metric for anthropometrics)

    The resulting dataset is ready for graph construction via
    :class:`~unravel.american_football.graphs.AmericanFootballGraphConverter`.

    Args:
        tracking_file_path (str): Path to tracking CSV file. Must contain columns:
            gameId, playId, nflId, frameId, x, y, s (speed), o (orientation),
            dir (direction), team (or club).
        players_file_path (str): Path to players CSV file. Must contain: nflId,
            position (or officialPosition), height, weight.
        plays_file_path (str): Path to plays CSV file. Must contain: gameId, playId,
            possessionTeam.
        sample_rate (float, optional): Sampling rate for downsampling frames. For example,
            0.5 keeps every 2nd frame. Defaults to None (no downsampling).
        max_player_speed (float, optional): Maximum physically plausible player speed (m/s)
            for filtering outliers. Defaults to 12.0 m/s (~27 mph).
        max_ball_speed (float, optional): Maximum physically plausible ball speed (m/s).
            Defaults to 28.0 m/s (~63 mph).
        max_player_acceleration (float, optional): Maximum player acceleration (m/s²).
            Defaults to 6.0 m/s².
        max_ball_acceleration (float, optional): Maximum ball acceleration (m/s²).
            Defaults to 13.5 m/s².
        orient_ball_owning (bool, optional): Whether to normalize coordinate system so
            the offense always attacks left-to-right. Defaults to True (recommended).
        **kwargs: Additional arguments passed to DefaultDataset.

    Attributes:
        data (pl.DataFrame): Processed tracking data with columns:
            - game_id, play_id, frame_id: Identifiers
            - object_id: Player NFL ID (or "football" for ball)
            - team_id: Team abbreviation or "football"
            - x, y: Position in yards (centered at midfield)
            - s: Speed in yards/second
            - o: Orientation angle in radians [-π, π]
            - dir: Direction of movement in radians [-π, π]
            - position_name: Player position (e.g., "QB", "WR", "CB")
            - height_cm: Player height in centimeters (rounded to nearest 10cm)
            - weight_kg: Player weight in kilograms (rounded to nearest 10kg)
            - ball_owning_team_id: Team with possession
        settings (DefaultSettings): Configuration object with pitch dimensions,
            orientation settings, and speed thresholds.

    Raises:
        NotImplementedError: If orient_ball_owning=False (currently unsupported).

    Example:
        >>> from unravel.american_football.dataset import BigDataBowlDataset
        >>>
        >>> # Load Big Data Bowl 2024 data
        >>> dataset = BigDataBowlDataset(
        ...     tracking_file_path="tracking_week_1.csv",
        ...     players_file_path="players.csv",
        ...     plays_file_path="plays.csv",
        ...     sample_rate=1.0,  # Use all frames
        ...     orient_ball_owning=True
        ... )
        >>>
        >>> # Access processed data
        >>> print(dataset.data)
        >>> print(f"Total frames: {dataset.data['frame_id'].n_unique()}")
        >>> print(f"Total plays: {dataset.data['play_id'].n_unique()}")
        >>>
        >>> # Downsample to 5 Hz (every other frame from 10 Hz)
        >>> dataset_5hz = BigDataBowlDataset(
        ...     tracking_file_path="tracking_week_1.csv",
        ...     players_file_path="players.csv",
        ...     plays_file_path="plays.csv",
        ...     sample_rate=0.5  # Keep every 2nd frame
        ... )
        >>>
        >>> # Add dummy labels for GNN training
        >>> dataset.add_dummy_labels()
        >>> dataset.add_graph_ids()

    Note:
        - Big Data Bowl data uses yards as the unit. The coordinate system is centered
          at midfield (x=0) with y=0 at the center of the field.
        - Player heights and weights are rounded to the nearest 10 cm / 10 kg to protect
          player privacy while retaining useful anthropometric information.
        - The orientation normalization (orient_ball_owning=True) ensures offensive
          players always attack from left to right, simplifying model training.
        - Frame IDs are computed as: play_id * 100,000 + frameId to ensure global uniqueness.
        - The "football" object has team_id="football" and is included in every frame.

    Warning:
        NFL Big Data Bowl data format can vary by year. This loader is tested on
        2023-2024 formats. Older competitions may require modifications.

    See Also:
        :class:`~unravel.american_football.graphs.AmericanFootballGraphConverter`:
            Convert to graph format for GNN training.
        :meth:`add_dummy_labels`: Add placeholder labels for testing.
        :meth:`add_graph_ids`: Add graph identifiers for batching.
        :doc:`../tutorials/american_football`: Tutorial on NFL tracking data analysis.
    """
    def __init__(
        self,
        tracking_file_path: str,
        players_file_path: str,
        plays_file_path: str,
        sample_rate: float = None,
        max_player_speed: float = 12.0,
        max_ball_speed: float = 28.0,
        max_player_acceleration: float = 6.0,
        max_ball_acceleration: float = 13.5,
        orient_ball_owning: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tracking_file_path = tracking_file_path
        self.players_file_path = players_file_path
        self.plays_file_path = plays_file_path
        self.sample_rate = 1 if sample_rate is None else sample_rate

        self._max_player_speed = max_player_speed
        self._max_ball_speed = max_ball_speed
        self._max_player_acceleration = max_player_acceleration
        self._max_ball_acceleration = max_ball_acceleration
        self._orient_ball_owning = orient_ball_owning

        self.load()

    def __apply_settings(
        self,
    ):
        return DefaultSettings(
            provider="nfl",
            home_team_id=None,
            away_team_id=None,
            pitch_dimensions=AmericanFootballPitchDimensions(),
            orientation=(
                Orientation.BALL_OWNING_TEAM
                if self._orient_ball_owning
                else Orientation.NOT_SET
            ),
            max_player_speed=self._max_player_speed,
            max_ball_speed=self._max_ball_speed,
            max_player_acceleration=self._max_player_acceleration,
            max_ball_acceleration=self._max_ball_acceleration,
            ball_carrier_threshold=None,
        )

    def load(self):
        self.settings = self.__apply_settings()

        pitch_length = self.settings.pitch_dimensions.pitch_length
        pitch_width = self.settings.pitch_dimensions.pitch_width

        sample = 1.0 / self.sample_rate

        df = pl.scan_csv(
            self.tracking_file_path,
            separator=",",
            encoding="utf8",
            null_values=["NA", "NULL", ""],
            try_parse_dates=True,
        )

        play_direction = "left"

        if "club" in df.collect_schema().names():
            df = df.rename({"club": Column.TEAM_ID})
        elif "team" in df.collect_schema().names():
            df = df.rename({"team": Column.TEAM_ID})

        if self._orient_ball_owning:
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
                            ((pl.col(Column.DIRECTION) * np.pi / 180) + np.pi)
                            % (2 * np.pi)
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
                        pl.when(pl.col(Column.TEAM_ID) == Constant.BALL)
                        .then(-9999.9)
                        .otherwise(pl.col("nflId"))
                        .alias("nflId"),
                    ]
                )
                .with_columns(
                    [
                        pl.lit(play_direction).alias("playDirection"),
                    ]
                )
                .filter((pl.col("frameId") % sample) == 0)
            ).collect()
        else:
            raise NotImplementedError(
                "Currently, BigDataBowlDataset only allows Orientation.BALL_OWNING"
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
            players = players.rename({"position": Column.POSITION_NAME})
        elif "officialPosition" in players.columns:
            players = players.rename({"officialPosition": Column.POSITION_NAME})

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
        ).rename(
            {
                "gameId": Column.GAME_ID,
                "playId": Column.PLAY_ID,
                "possessionTeam": Column.BALL_OWNING_TEAM_ID,
            }
        )

        df = df.join(
            (
                players.select(
                    [
                        "nflId",
                        Column.POSITION_NAME,
                        Column.HEIGHT_CM,
                        Column.WEIGHT_KG,
                    ]
                )
            ),
            on="nflId",
            how="left",
        )

        df = df.rename(
            {
                "nflId": Column.OBJECT_ID,
                "gameId": Column.GAME_ID,
                "playId": Column.PLAY_ID,
                "s": Column.SPEED,
            }
        )

        df = df.join(
            (plays.select(Group.BY_PLAY_BALL_OWNING)),
            on=[Column.GAME_ID, Column.PLAY_ID],
            how="left",
        )

        df = df.with_columns(
            [
                (pl.col(Column.PLAY_ID) * 100_000 + pl.col("frameId")).alias(
                    Column.FRAME_ID
                )
            ]
        ).drop(["frameId"])

        self.data = df.sort(
            [Column.GAME_ID, Column.PLAY_ID, Column.FRAME_ID, Column.OBJECT_ID]
        )

        # update pitch dimensions to how it looks after loading
        self.settings.pitch_dimensions = AmericanFootballPitchDimensions(
            x_dim=Dimension(min=-pitch_length / 2, max=pitch_length / 2),
            y_dim=Dimension(min=-pitch_width / 2, max=pitch_width / 2),
            standardized=False,
            unit=Unit.YARDS,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

        return self.data, self.settings

    def add_dummy_labels(
        self, by: List[str] = [Column.GAME_ID, Column.FRAME_ID]
    ) -> pl.DataFrame:
        self.data = add_dummy_label_column(self.data, by, self._label_column)
        return self.data

    def add_graph_ids(self, by: List[str] = [Column.GAME_ID]) -> pl.DataFrame:
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
        # Convert height and weight to centimeters and kilograms
        # Round them to 0.1 to make sure we don't leak any player specific info
        df = (
            df.with_columns(
                [
                    ((pl.col("feet") * 30.48 + pl.col("inches") * 2.54) / 10)
                    .round(0)
                    .alias(Column.HEIGHT_CM),
                    ((pl.col("weight") * 0.453592) / 10)
                    .round(0)
                    .alias(Column.WEIGHT_KG),
                ]
            )
            .with_columns(
                [
                    (pl.col(Column.HEIGHT_CM) * 10).alias(Column.HEIGHT_CM),
                    (pl.col(Column.WEIGHT_KG) * 10).alias(Column.WEIGHT_KG),
                ]
            )
            .drop(["height", "feet", "inches", "weight"])
        )
        return df
