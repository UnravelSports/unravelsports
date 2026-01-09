from kloppy.domain import (
    TrackingDataset,
    Frame,
    Orientation,
    DatasetTransformer,
    DatasetFlag,
    SecondSpectrumCoordinateSystem,
    MetricPitchDimensions,
    Provider,
)

from typing import List, Dict, Union, Literal, Tuple, Optional

from dataclasses import field, dataclass

from ...utils import (
    DefaultDataset,
    DefaultSettings,
    add_dummy_label_column,
    add_graph_id_column,
)

from .objects import Column, Group, Constant
from .utils import apply_speed_acceleration_filters

import polars as pl

import warnings


DEFAULT_PLAYER_SMOOTHING_PARAMS = {"window_length": 7, "polyorder": 1}
DEFAULT_BALL_SMOOTHING_PARAMS = {"window_length": 3, "polyorder": 1}


@dataclass
class SoccerObject:
    """Represents a player or ball object in soccer tracking data.

    This dataclass stores metadata about players and the ball, including
    identification, team affiliation, and position information.

    Attributes:
        id: Unique identifier for the object (player ID or 'ball').
        team_id: Team identifier the object belongs to.
        position_name: Position code (e.g., 'GK', 'CB', 'LW') or 'ball'.
        number: Jersey number for players. Defaults to None.
        name: Player name. Defaults to None.
        team_name: Name of the team. Defaults to None.
        is_gk: Whether the player is a goalkeeper. Defaults to None.
        is_home: Whether the player is on the home team. Defaults to None.
        object_type: Type of object, either 'ball' or 'player'. Defaults to 'player'.
    """

    id: Union[str, int]
    team_id: Union[str, int]
    position_name: str
    number: int = None
    name: str = None
    team_name: str = None
    is_gk: bool = None
    is_home: bool = None
    object_type: Literal["ball", "player"] = "player"

    def __repr__(self):
        return f"({self.object_type.capitalize()} name={self.name}, number={self.number}, player_id={self.id}, is_gk={self.is_gk}, is_home={self.is_home})"


@dataclass
class KloppyPolarsDataset(DefaultDataset):
    """Convert Kloppy soccer tracking data to Polars DataFrame format.

    This class takes tracking data loaded via Kloppy (supporting providers like Sportec,
    SkillCorner, Tracab, SecondSpectrum, etc.) and converts it into a fast, efficient
    Polars DataFrame with computed velocities, accelerations, and ball carrier inference.

    The conversion process includes:
    - Coordinate system standardization
    - Velocity and acceleration computation with optional smoothing
    - Ball carrier and ball owning team inference
    - Goalkeeper position identification
    - Speed and acceleration filtering to remove outliers
    - Optional orientation normalization (attacking left-to-right)

    Args:
        kloppy_dataset: A Kloppy TrackingDataset instance containing the raw tracking data.
        ball_carrier_threshold: Maximum distance (in meters) between player and ball to
            be considered the ball carrier. Defaults to 25.0.
        max_player_speed: Maximum realistic player speed in m/s. Values above this are
            capped to prevent sensor errors. Defaults to 12.0 m/s.
        max_ball_speed: Maximum realistic ball speed in m/s. Values above this are
            capped. Defaults to 28.0 m/s.
        max_player_acceleration: Maximum realistic player acceleration in m/s². Values
            above this are capped. Defaults to 6.0 m/s².
        max_ball_acceleration: Maximum realistic ball acceleration in m/s². Values above
            this are capped. Defaults to 13.5 m/s².
        orient_ball_owning: If True, normalize coordinates so the team with possession
            always attacks from left to right. Defaults to True.
        add_smoothing: If True, apply Savitzky-Golay smoothing to velocities to reduce
            noise. Defaults to True.
        **kwargs: Additional keyword arguments passed to DefaultDataset.

    Attributes:
        data (pl.DataFrame): The converted Polars DataFrame with all tracking data.
        settings (DefaultSettings): Configuration and metadata for the dataset.
        home_players (List[SoccerObject]): List of home team player objects.
        away_players (List[SoccerObject]): List of away team player objects.
        kloppy_dataset (TrackingDataset): The original Kloppy dataset.

    Raises:
        Exception: If kloppy_dataset is not a TrackingDataset instance.
        Exception: If ball_carrier_threshold is not a float.
        ValueError: If the dataset orientation is NOT_SET.
        ValueError: If ball owning team must be inferred but ball_carrier_threshold is None.

    Example:
        >>> from kloppy import sportec
        >>> from unravel.soccer import KloppyPolarsDataset
        >>>
        >>> # Load tracking data with Kloppy
        >>> kloppy_dataset = sportec.load_open_tracking_data(only_alive=True)
        >>>
        >>> # Convert to Polars format
        >>> polars_dataset = KloppyPolarsDataset(
        ...     kloppy_dataset=kloppy_dataset,
        ...     ball_carrier_threshold=25.0,
        ...     max_player_speed=12.0,
        ...     orient_ball_owning=True
        ... )
        >>>
        >>> # Access the DataFrame
        >>> df = polars_dataset.data
        >>> print(df.head())
        >>>
        >>> # Add dummy labels for training
        >>> polars_dataset.add_dummy_labels(by=["frame_id"])
        >>>
        >>> # Add graph IDs for grouping
        >>> polars_dataset.add_graph_ids(by=["frame_id"])

    Note:
        For non-Sportec providers, always use ``only_alive=True`` or
        ``include_empty_frames=False`` when loading data with Kloppy to avoid
        frames without ball tracking data.

    Warning:
        If the dataset doesn't include ball owning team information, it will be
        inferred using distance to ball. This may cause unexpected results in
        situations where the ball is contested or in the air.

    See Also:
        :class:`~unravel.soccer.SoccerGraphConverter`: Convert to graph structures.
        :func:`~unravel.utils.add_dummy_label_column`: Add labels for training.
        :func:`~unravel.utils.add_graph_id_column`: Add graph IDs for grouping.
    """

    def __init__(
        self,
        kloppy_dataset: TrackingDataset,
        ball_carrier_threshold: float = 25.0,
        max_player_speed: float = 12.0,
        max_ball_speed: float = 28.0,
        max_player_acceleration: float = 6.0,
        max_ball_acceleration: float = 13.5,
        orient_ball_owning: bool = True,
        add_smoothing: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kloppy_dataset = kloppy_dataset

        self._ball_carrier_threshold = ball_carrier_threshold
        self._max_player_speed = max_player_speed
        self._max_ball_speed = max_ball_speed
        self._max_player_acceleration = max_player_acceleration
        self._max_ball_acceleration = max_ball_acceleration
        self._orient_ball_owning = orient_ball_owning
        self._infer_goalkeepers: bool = False
        self._add_smoothing: bool = add_smoothing

        if not isinstance(self.kloppy_dataset, TrackingDataset):
            raise Exception("'kloppy_dataset' should be of type float")

        if not isinstance(self._ball_carrier_threshold, float):
            raise Exception("'ball_carrier_threshold' should be of type float")

        self.load()

    def __repr__(self) -> str:
        n_frames = (
            self.data[Column.FRAME_ID].n_unique() if hasattr(self, "data") else None
        )
        return f"KloppyPolarsDataset(n_frames={n_frames})"

    def __transform_orientation(
        self,
    ) -> Tuple[TrackingDataset, Union[None, TrackingDataset]]:
        """
        We create orientation transformed kloppy datasets.
        We set it to Orientation.STATIC_HOME_AWAY if it is currently BALL_OWNING to compute speed and accelerations correctly using Polars.
        If we set it Orientation.BALL_OWNING directly, as we did previously, the coordinates can flip by *-1.0 in the middle of a sequence, this breaks the
        speed and acceleration computations.

        We flip it to BALL_OWNING later using __fix_orientation_to_ball_owning, if needed

        We keep the provided kloppy orientation if we set orient_ball_owning to False
        """
        secondspectrum_coordinate_system = SecondSpectrumCoordinateSystem(
            pitch_length=self.kloppy_dataset.metadata.pitch_dimensions.pitch_length,
            pitch_width=self.kloppy_dataset.metadata.pitch_dimensions.pitch_width,
        )

        kloppy_static = DatasetTransformer.transform_dataset(
            dataset=self.kloppy_dataset,
            to_coordinate_system=secondspectrum_coordinate_system,
            to_orientation=Orientation.STATIC_HOME_AWAY,
        )

        return kloppy_static

    def __get_objects(self):
        def __artificial_game_id() -> str:
            from uuid import uuid4

            return str(uuid4())

        home_team, away_team = self.kloppy_dataset.metadata.teams

        if all(
            item is None for item in [p.starting_position for p in home_team.players]
        ):
            self._infer_goalkeepers = True
            home_players = [
                SoccerObject(
                    id=p.player_id,
                    team_id=p.team.team_id,
                    position_name=None,
                    number=p.jersey_no,
                    name=p.last_name,
                    team_name=p.team.name,
                    is_home=True,
                    object_type="player",
                )
                for p in home_team.players
            ]
            away_players = [
                SoccerObject(
                    id=p.player_id,
                    team_id=p.team.team_id,
                    position_name=None,
                    number=p.jersey_no,
                    name=p.last_name,
                    team_name=p.team.name,
                    is_home=False,
                    object_type="player",
                )
                for p in away_team.players
            ]
        else:
            home_players = [
                SoccerObject(
                    id=p.player_id,
                    team_id=p.team.team_id,
                    position_name=p.starting_position.code,
                    number=p.jersey_no,
                    name=p.last_name,
                    team_name=p.team.name,
                    is_home=True,
                    is_gk=True if p.starting_position.code == "GK" else False,
                    object_type="player",
                )
                for p in home_team.players
            ]
            away_players = [
                SoccerObject(
                    id=p.player_id,
                    team_id=p.team.team_id,
                    position_name=p.starting_position.code,
                    number=p.jersey_no,
                    name=p.last_name,
                    team_name=p.team.name,
                    is_home=False,
                    is_gk=True if p.starting_position.code == "GK" else False,
                    object_type="player",
                )
                for p in away_team.players
            ]
        ball_object = SoccerObject(Constant.BALL, Constant.BALL, Constant.BALL)
        game_id = self.kloppy_dataset.metadata.game_id
        if game_id is None:
            game_id = __artificial_game_id()
        return (home_players, away_players, ball_object, game_id)

    def __unpivot(self, df, object, coordinate):
        column = f"{object.id}_{coordinate}"

        return df.unpivot(
            index=[
                Column.PERIOD_ID,
                Column.TIMESTAMP,
                Column.FRAME_ID,
                Column.BALL_STATE,
                Column.BALL_OWNING_TEAM_ID,
            ],  # Columns to keep
            on=[column],
            value_name=coordinate,
            variable_name=Column.OBJECT_ID,
        ).with_columns(
            pl.col(Column.OBJECT_ID).str.replace(
                f"_{coordinate}", ""
            )  # Remove the coordinate suffix
        )

    def __apply_smoothing(self, df: pl.DataFrame, smoothing_params: dict):
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError(
                "Seems like you don't have scipy installed. Please"
                " install it using: pip install scipy"
            )

        if not smoothing_params.get("window_length"):
            raise ValueError(
                "Missing parameter 'window_length' in player_smoothing_params and/or ball_smoothing_params"
            )
        if not smoothing_params.get("polyorder"):
            raise ValueError(
                "Missing parameter 'polyorder' in player_smoothing_params and/or ball_smoothing_params"
            )

        vx_smooth = f"{Column.VX}_smoothed"
        vy_smooth = f"{Column.VY}_smoothed"
        vz_smooth = f"{Column.VZ}_smoothed"

        # DEBUG: Check group sizes
        group_sizes = df.group_by(Group.BY_OBJECT_PERIOD).agg(
            pl.col(Column.VX).count().alias("count")
        )

        window_length = smoothing_params["window_length"]
        polyorder = smoothing_params["polyorder"]

        def apply_savgol(series):
            """Apply savgol filter to a series (array of values)."""
            values = series.to_numpy()
            if len(values) < window_length:
                return values.tolist()
            return savgol_filter(
                values,
                window_length=window_length,
                polyorder=polyorder,
            ).tolist()

        smoothed = df.group_by(Group.BY_OBJECT_PERIOD, maintain_order=True).agg(
            [
                pl.col(Column.VX)
                .map_batches(
                    apply_savgol, return_dtype=pl.List(pl.Float64), returns_scalar=True
                )
                .alias(vx_smooth),
                pl.col(Column.VY)
                .map_batches(
                    apply_savgol, return_dtype=pl.List(pl.Float64), returns_scalar=True
                )
                .alias(vy_smooth),
                pl.col(Column.VZ)
                .map_batches(
                    apply_savgol, return_dtype=pl.List(pl.Float64), returns_scalar=True
                )
                .alias(vz_smooth),
            ]
        )
        # Explode the smoothed columns back to original shape
        smoothed_exploded = smoothed.explode([vx_smooth, vy_smooth, vz_smooth])
        # Combine with the original DataFrame if needed
        return df.with_columns(
            vx=smoothed_exploded[vx_smooth],
            vy=smoothed_exploded[vy_smooth],
            vz=smoothed_exploded[vz_smooth],
        )

    def __add_velocity(
        self,
        df: pl.DataFrame,
        player_smoothing_params: dict,
        ball_smoothing_params: dict,
    ):
        df = (
            df.sort(
                Group.BY_OBJECT_PERIOD + [Column.TIMESTAMP, Column.TEAM_ID],
                nulls_last=True,
            )
            .with_columns(
                [
                    # Calculate differences within each group
                    pl.col(Column.X).diff().over(Group.BY_OBJECT_PERIOD).alias("dx"),
                    pl.col(Column.Y).diff().over(Group.BY_OBJECT_PERIOD).alias("dy"),
                    pl.col(Column.Z).diff().over(Group.BY_OBJECT_PERIOD).alias("dz"),
                    (pl.col(Column.TIMESTAMP).dt.total_milliseconds() / 1_000)
                    .diff()
                    .over(Group.BY_OBJECT_PERIOD)
                    .alias("dt"),
                ]
            )
            .with_columns(
                [
                    # Compute velocity components
                    (pl.col("dx") / pl.col("dt")).alias(Column.VX),
                    (pl.col("dy") / pl.col("dt")).alias(Column.VY),
                    (pl.col("dz") / pl.col("dt")).alias(Column.VZ),
                ]
            )
            .with_columns(
                [
                    # Fill null values in vx and vy
                    pl.col(Column.VX).fill_null(0).alias(Column.VX),
                    pl.col(Column.VY).fill_null(0).alias(Column.VY),
                    pl.col(Column.VZ).fill_null(0).alias(Column.VZ),
                ]
            )
        )

        if self._add_smoothing and player_smoothing_params:
            player_df = self.__apply_smoothing(
                df=df.filter(pl.col(Column.OBJECT_ID) != self._ball_object.id),
                smoothing_params=player_smoothing_params,
            )
        else:
            player_df = df.filter(pl.col(Column.OBJECT_ID) != self._ball_object.id)

        if self._add_smoothing and ball_smoothing_params:
            ball_df = self.__apply_smoothing(
                df.filter(pl.col(Column.OBJECT_ID) == self._ball_object.id),
                smoothing_params=ball_smoothing_params,
            )
        else:
            ball_df = df.filter(pl.col(Column.OBJECT_ID) == self._ball_object.id)

        df = pl.concat([player_df, ball_df])
        df = df.with_columns(
            [
                (
                    pl.col(Column.VX) ** 2
                    + pl.col(Column.VY) ** 2
                    + pl.col(Column.VZ) ** 2
                )
                .sqrt()
                .alias(Column.SPEED)
            ]
        )
        return df

    def __add_acceleration(self, df: pl.DataFrame):
        return (
            df.with_columns(
                [
                    # Calculate differences in vx, vy, and dt for acceleration
                    pl.col(Column.VX).diff().over(Group.BY_OBJECT_PERIOD).alias("dvx"),
                    pl.col(Column.VY).diff().over(Group.BY_OBJECT_PERIOD).alias("dvy"),
                    pl.col(Column.VZ).diff().over(Group.BY_OBJECT_PERIOD).alias("dvz"),
                ]
            )
            .with_columns(
                [
                    # Compute ax and ay
                    (pl.col("dvx") / pl.col("dt")).alias(Column.AX),
                    (pl.col("dvy") / pl.col("dt")).alias(Column.AY),
                    (pl.col("dvz") / pl.col("dt")).alias(Column.AZ),
                ]
            )
            .with_columns(
                [
                    # Fill null values in vx and vy
                    pl.col(Column.AX).fill_null(0).alias(Column.AX),
                    pl.col(Column.AY).fill_null(0).alias(Column.AY),
                    pl.col(Column.AZ).fill_null(0).alias(Column.AZ),
                ]
            )
            .with_columns(
                [
                    # Compute magnitude of acceleration a
                    (
                        pl.col(Column.AX) ** 2
                        + pl.col(Column.AY) ** 2
                        + pl.col(Column.AZ) ** 2
                    )
                    .sqrt()
                    .alias(Column.ACCELERATION)
                ]
            )
        )

    def __melt(
        self,
        df: pl.DataFrame,
        home_players: List[SoccerObject],
        away_players: List[SoccerObject],
        ball_object: SoccerObject,
        game_id: Union[int, str],
    ):
        melted_dfs = []
        columns = df.columns

        for object in [ball_object] + home_players + away_players:
            melted_object_dfs = []
            for k, coordinate in enumerate([Column.X, Column.Y, Column.Z]):
                if object.id != Constant.BALL and coordinate == Column.Z:
                    continue
                if not any(
                    object.id + "_" + coordinate == column for column in columns
                ):
                    continue

                melted_df = self.__unpivot(df, object, coordinate)

                if object.id == Constant.BALL and coordinate == Column.Z:
                    if melted_df[coordinate].is_null().all():
                        melted_df = melted_df.with_columns(
                            [pl.lit(0.0).alias(Column.Z)]
                        )
                if k == 0:
                    melted_object_dfs.append(melted_df)
                else:
                    melted_object_dfs.append(melted_df[[coordinate]])

            if melted_object_dfs:
                object_df = pl.concat(melted_object_dfs, how="horizontal")
                if Column.Z not in object_df.columns:
                    object_df = object_df.with_columns([pl.lit(0.0).alias(Column.Z)])
                object_df = object_df.with_columns(
                    [
                        pl.lit(object.team_id).cast(pl.Utf8).alias(Column.TEAM_ID),
                        pl.lit(object.position_name).alias(Column.POSITION_NAME),
                    ]
                )

                melted_dfs.append(object_df)

        df = pl.concat(melted_dfs, how="vertical")
        df = df.with_columns([pl.lit(game_id).alias(Column.GAME_ID)])
        df = df.sort(
            by=[Column.PERIOD_ID, Column.TIMESTAMP, Column.TEAM_ID], nulls_last=True
        )
        return df

    def __infer_ball_carrier(self, df: pl.DataFrame):
        if Column.BALL_OWNING_PLAYER_ID not in df.columns:
            df = df.with_columns(
                pl.lit(None)
                .cast(df.schema[Column.OBJECT_ID])
                .alias(Column.BALL_OWNING_PLAYER_ID)
            )
        # handle the non ball owning frames
        ball = df.filter(pl.col(Column.TEAM_ID) == Constant.BALL)
        players = df.filter(pl.col(Column.TEAM_ID) != Constant.BALL)

        # ball owning team is empty, so we can drop it. Goal is to replace it
        players_ball = players.join(
            ball.select(
                Group.BY_FRAME
                + [
                    pl.col(Column.X).alias("ball_x"),
                    pl.col(Column.Y).alias("ball_y"),
                    pl.col(Column.Z).alias("ball_z"),
                ]
            ),
            on=Group.BY_FRAME,
            how="left",
        ).with_columns(
            [
                (
                    (pl.col(Column.X) - pl.col("ball_x")) ** 2
                    + (pl.col(Column.Y) - pl.col("ball_y")) ** 2
                    + (pl.col(Column.Z) - pl.col("ball_z")) ** 2
                )
                .sqrt()
                .alias("ball_dist")
            ]
        )
        # Update ball_owning_team if necessary
        ball_owning_team = (players_ball.drop(Column.BALL_OWNING_TEAM_ID)).join(
            players_ball.group_by(Group.BY_FRAME, maintain_order=True)
            .agg(
                [
                    pl.when((pl.col(Column.BALL_OWNING_TEAM_ID).is_null()))
                    .then(
                        pl.col(Column.TEAM_ID)
                        .filter(
                            (pl.col("ball_dist") == pl.col("ball_dist").min())
                            & (
                                pl.col("ball_dist").min()
                                < self.settings.ball_carrier_threshold
                            )
                        )
                        .first()
                    )
                    .otherwise(pl.col(Column.BALL_OWNING_TEAM_ID))
                    .alias(Column.BALL_OWNING_TEAM_ID),
                ]
            )
            .with_columns(
                [
                    pl.col(Column.BALL_OWNING_TEAM_ID)
                    .list.first()
                    .alias(Column.BALL_OWNING_TEAM_ID),
                ]
            ),
            on=Group.BY_FRAME,
            how="left",
        )
        # Make sure the ball owning player is on the ball owning team
        result = (
            (ball_owning_team.drop(Column.BALL_OWNING_PLAYER_ID))
            .join(
                ball_owning_team.filter(
                    (pl.col(Column.BALL_OWNING_TEAM_ID) == pl.col(Column.TEAM_ID))
                )
                .group_by(Group.BY_FRAME, maintain_order=True)
                .agg(
                    [
                        pl.when((pl.col(Column.BALL_OWNING_PLAYER_ID).is_null()))
                        .then(
                            pl.col(Column.OBJECT_ID)
                            .filter(
                                (pl.col("ball_dist") == pl.col("ball_dist").min())
                                & (
                                    pl.col("ball_dist").min()
                                    < self.settings.ball_carrier_threshold
                                )
                            )
                            .first()
                        )
                        .otherwise(pl.col(Column.BALL_OWNING_PLAYER_ID))
                        .alias(Column.BALL_OWNING_PLAYER_ID)
                    ]
                )
                .with_columns(
                    [
                        pl.col(Column.BALL_OWNING_PLAYER_ID)
                        .list.first()
                        .alias(Column.BALL_OWNING_PLAYER_ID),
                    ]
                ),
                on=Group.BY_FRAME,
                how="left",
            )
            .select(
                Group.BY_FRAME
                + [Column.BALL_OWNING_TEAM_ID, Column.BALL_OWNING_PLAYER_ID]
            )
            .unique()
        )
        df = (
            df.drop([Column.BALL_OWNING_PLAYER_ID, Column.BALL_OWNING_TEAM_ID])
            .join(result, how="left", on=Group.BY_FRAME)
            .with_columns(
                pl.when(
                    pl.col(Column.OBJECT_ID) == pl.col(Column.BALL_OWNING_PLAYER_ID)
                )
                .then(True)
                .otherwise(False)
                .alias(Column.IS_BALL_CARRIER)
            )
            .drop(Column.BALL_OWNING_PLAYER_ID)
            .drop_nulls(subset=Column.BALL_OWNING_TEAM_ID)
        )
        return df

    def __infer_goalkeepers(self, df: pl.DataFrame):
        goal_x = self.settings.pitch_dimensions.pitch_length / 2
        goal_y = 0

        df_with_distances = df.filter(
            pl.col(Column.TEAM_ID) != Constant.BALL
        ).with_columns(
            [
                ((pl.col(Column.X) - (-goal_x)) ** 2 + (pl.col(Column.Y) - goal_y) ** 2)
                .sqrt()
                .alias("dist_left"),
                ((pl.col(Column.X) - goal_x) ** 2 + (pl.col(Column.Y) - goal_y) ** 2)
                .sqrt()
                .alias("dist_right"),
            ]
        )
        result = (
            df_with_distances.with_columns(
                [
                    pl.col("dist_left")
                    .min()
                    .over(Group.BY_FRAME_TEAM)
                    .alias("min_dist_left"),
                    pl.col("dist_right")
                    .min()
                    .over(Group.BY_FRAME_TEAM)
                    .alias("min_dist_right"),
                ]
            )
            .with_columns(
                [
                    pl.when(
                        pl.col(Column.TEAM_ID) == pl.col(Column.BALL_OWNING_TEAM_ID)
                    )
                    .then(
                        pl.when(pl.col("dist_left") == pl.col("min_dist_left"))
                        .then(pl.lit("GK"))
                        .otherwise(None)
                    )
                    .otherwise(
                        pl.when(pl.col("dist_right") == pl.col("min_dist_right"))
                        .then(pl.lit("GK"))
                        .otherwise(None)
                    )
                    .alias("position_name")
                ]
            )
            .drop(["min_dist_left", "min_dist_right", "dist_left", "dist_right"])
        )
        ball_rows = df.filter(pl.col(Column.TEAM_ID) == Constant.BALL)
        non_ball_rows = result

        return pl.concat([ball_rows, non_ball_rows], how="vertical").sort(
            Group.BY_FRAME_TEAM
        )

    def convert_orientation_to_ball_owning(self, df: pl.DataFrame):
        """Convert field orientation so attacking team always goes left-to-right.

        This method normalizes the coordinate system so that the team with possession
        always attacks from left to right, regardless of which half they're in. This
        helps machine learning models by providing consistent attacking directionality.

        When the away team has possession, all spatial coordinates (x, y) and their
        derivatives (vx, vy, ax, ay) are multiplied by -1.

        Args:
            df: The DataFrame with STATIC_HOME_AWAY orientation.

        Returns:
            pl.DataFrame: DataFrame with BALL_OWNING_TEAM orientation.

        Raises:
            ValueError: If orientation is already BALL_OWNING_TEAM.

        Example:
            >>> # Typically called automatically if orient_ball_owning=True
            >>> # But can be called manually:
            >>> df = dataset.convert_orientation_to_ball_owning(dataset.data)

        Note:
            This is called automatically during ``load()`` if the ``orient_ball_owning``
            parameter is set to True in ``__init__``.

            The following columns are flipped when away team has possession:
            - x, y: Position coordinates
            - vx, vy: Velocity components
            - ax, ay: Acceleration components

        See Also:
            Kloppy Orientation documentation for more details on coordinate systems.
        """
        # When orient_ball_owning is True, it means the orientation has to flip from "STATIC_HOME_AWAY" to "BALL_OWNING" in the Polars dataframe
        # This means that when away is the attacking team we can flip all coordinates by -1.0
        if self.settings.orientation == Orientation.BALL_OWNING_TEAM:
            raise ValueError(
                "Orientation is already BALL_OWNING_TEAM this operation is not possible..."
            )

        flip_columns = [Column.X, Column.Y, Column.VX, Column.VY, Column.AX, Column.AY]

        self.settings.orientation = Orientation.BALL_OWNING_TEAM

        home_team, _ = self.kloppy_dataset.metadata.teams
        return df.with_columns(
            [
                pl.when(
                    pl.col(Column.BALL_OWNING_TEAM_ID).cast(str)
                    != str(home_team.team_id)
                )
                .then(pl.col(flip_columns) * -1)
                .otherwise(pl.col(flip_columns))
            ]
        )

    def __apply_settings(
        self,
        pitch_dimensions,
    ):
        home_team, away_team = self.kloppy_dataset.metadata.teams
        return DefaultSettings(
            provider="secondspectrum",
            orientation=self.kloppy_dataset.metadata.orientation,
            home_team_id=home_team.team_id,
            away_team_id=away_team.team_id,
            players=[
                {
                    "player_id": p.player_id,
                    "team_id": p.team.team_id,
                    "player": p.full_name,
                    "team": p.team.name,
                    "jersey_no": p.jersey_no,
                }
                for p in home_team.players + away_team.players
            ],
            pitch_dimensions=pitch_dimensions,
            max_player_speed=self._max_player_speed,
            max_ball_speed=self._max_ball_speed,
            max_player_acceleration=self._max_player_acceleration,
            max_ball_acceleration=self._max_ball_acceleration,
            ball_carrier_threshold=self._ball_carrier_threshold,
            frame_rate=self.kloppy_dataset.metadata.frame_rate,
        )

    def load(
        self,
    ):
        """Load and process the Kloppy tracking dataset into Polars DataFrame.

        This method performs the complete data transformation pipeline:

        1. Transform coordinate system to SecondSpectrum standard
        2. Extract player and ball metadata
        3. Convert wide format (columns per player) to long format
        4. Compute velocities with optional Savitzky-Golay smoothing
        5. Compute accelerations
        6. Filter unrealistic speed/acceleration values
        7. Infer ball carrier and ball owning team (if not provided)
        8. Optionally normalize orientation to ball-owning team
        9. Infer goalkeeper positions (if position data unavailable)

        The resulting DataFrame is stored in ``self.data`` and contains columns:
        - period_id, timestamp, frame_id: Temporal identifiers
        - id, team_id, position_name: Object identifiers
        - x, y, z: Positions
        - vx, vy, vz, speed: Velocities
        - ax, ay, az, acceleration: Accelerations
        - ball_state: Ball in/out of play
        - ball_owning_team_id: Team with possession
        - is_ball_carrier: Boolean flag for ball carrier
        - game_id: Match identifier

        Returns:
            KloppyPolarsDataset: Self, for method chaining.

        Raises:
            ValueError: If dataset orientation is NOT_SET.
            ValueError: If ball owning team inference is needed but
                ball_carrier_threshold is None.

        Example:
            >>> # Typically called automatically in __init__
            >>> # But can be called manually to reload:
            >>> dataset.load()

        Note:
            This method is called automatically during ``__init__``, so you
            typically don't need to call it manually unless reloading data.

        Warning:
            If ball owning team is not provided in the data, it will be inferred
            using distance thresholds, which may be inaccurate during contested
            ball situations.
        """
        if self.kloppy_dataset.metadata.orientation == Orientation.NOT_SET:
            raise ValueError(
                "Data sources with an undefined orientation can not be used inside the 'unravelsports' package..."
            )

        self.kloppy_dataset = self.__transform_orientation()

        self.settings = self.__apply_settings(
            pitch_dimensions=self.kloppy_dataset.metadata.pitch_dimensions
        )

        (self.home_players, self.away_players, self._ball_object, self._game_id) = (
            self.__get_objects()
        )

        df = self.kloppy_dataset.to_df(engine="polars")
        df = self.__melt(
            df, self.home_players, self.away_players, self._ball_object, self._game_id
        )
        df = self.__add_velocity(
            df, DEFAULT_PLAYER_SMOOTHING_PARAMS, DEFAULT_BALL_SMOOTHING_PARAMS
        )
        df = self.__add_acceleration(df)
        df = apply_speed_acceleration_filters(
            df,
            max_player_speed=self.settings.max_player_speed,
            max_ball_speed=self.settings.max_ball_speed,
            max_player_acceleration=self.settings.max_player_acceleration,
            max_ball_acceleration=self.settings.max_ball_acceleration,
        )
        df = df.drop(["dx", "dy", "dz", "dt", "dvx", "dvy", "dvz"])
        df = df.filter(~(pl.col(Column.X).is_null() & pl.col(Column.Y).is_null()))

        if df[Column.BALL_OWNING_TEAM_ID].is_null().all():
            if self._ball_carrier_threshold is None:
                raise ValueError(
                    f"This dataset requires us to infer the {Column.BALL_OWNING_TEAM_ID}, please specifiy a ball_carrier_threshold (float) to do so."
                )
            else:
                warnings.warn(
                    "This dataset does not come with 'ball owning team' information. As a result we infer this using distance to ball using the 'ball_carrier_threshold'. Please note this might cause unexpected results.",
                    UserWarning,
                )

        df = self.__infer_ball_carrier(df)

        if (
            self._orient_ball_owning
            and self.settings.orientation != Orientation.BALL_OWNING_TEAM
        ):
            df = self.convert_orientation_to_ball_owning(df)

        if self._infer_goalkeepers:
            df = self.__infer_goalkeepers(df)

        self.data = df.unique(
            [Column.OBJECT_ID, Column.FRAME_ID, Column.PERIOD_ID]
        ).sort([Column.FRAME_ID, Column.PERIOD_ID, Column.OBJECT_ID])
        return self

    def add_dummy_labels(
        self, by: List[str] = ["game_id", "frame_id"], random_seed: Optional[int] = None
    ) -> pl.DataFrame:
        """Add a column of random binary labels for testing/demonstration purposes.

        This method adds a 'label' column with random 0/1 values to the dataset.
        Useful for testing graph neural network pipelines before you have real labels.

        Args:
            by: Column names to group by before assigning labels. Each unique
                combination gets the same random label. Defaults to ["game_id", "frame_id"].
            random_seed: Random seed for reproducibility. If None, labels will be
                different each time. Defaults to None.

        Returns:
            pl.DataFrame: The updated DataFrame with 'label' column added.

        Example:
            >>> # Add random labels, one per frame
            >>> dataset.add_dummy_labels(by=["frame_id"])
            >>>
            >>> # Add labels grouped by possession
            >>> dataset.add_dummy_labels(by=["ball_owning_team_id", "period_id"])
            >>>
            >>> # Reproducible labels
            >>> dataset.add_dummy_labels(by=["frame_id"], random_seed=42)

        Note:
            In real applications, replace this with actual labels from your data:

            >>> import polars as pl
            >>> labels = pl.DataFrame({"frame_id": [...], "label": [...]})
            >>> dataset.data = dataset.data.join(labels, on="frame_id")

        See Also:
            :func:`~unravel.utils.add_dummy_label_column`: Underlying utility function.
        """
        self.data = add_dummy_label_column(
            self.data, by, self._label_column, random_seed
        )
        return self.data

    def add_graph_ids(self, by: List[str] = ["game_id", "period_id"]) -> pl.DataFrame:
        """Add a graph_id column for grouping frames into graph samples.

        This method adds a 'graph_id' column that groups tracking frames into
        distinct graph samples for GNN training. This is crucial for proper
        train/test splitting to avoid data leakage.

        Args:
            by: Column names to group by. Each unique combination gets a unique
                graph_id. Defaults to ["game_id", "period_id"].

        Returns:
            pl.DataFrame: The updated DataFrame with 'graph_id' column added.

        Example:
            >>> # Each frame is a separate graph
            >>> dataset.add_graph_ids(by=["frame_id"])
            >>>
            >>> # Group by possession (all frames in same possession = one graph)
            >>> dataset.add_graph_ids(by=["ball_owning_team_id", "period_id"])
            >>>
            >>> # Group by 10-frame sequences
            >>> dataset.data = dataset.data.with_columns(
            ...     (pl.col("frame_id") // 10).alias("sequence_id")
            ... )
            >>> dataset.add_graph_ids(by=["sequence_id"])

        Important:
            When splitting data for training, **always split by graph_id** to avoid
            data leakage. Never split by row index:

            >>> # CORRECT: Split by graph_id
            >>> train, test, val = dataset.split_test_train_validation(4, 1, 1)
            >>>
            >>> # WRONG: Don't split by index
            >>> train = dataset[:800]  # May have same game in train and test!

        See Also:
            :func:`~unravel.utils.add_graph_id_column`: Underlying utility function.
            :meth:`~unravel.utils.GraphDataset.split_test_train_validation`: Splitting method.
        """
        self.data = add_graph_id_column(self.data, by, self._graph_id_column)
        return self.data

    def get_player_by_id(self, player_id):
        if hasattr(self, "home_players") and hasattr(self, "away_players"):
            for player in self.home_players + self.away_players:
                if player.id == player_id:
                    return player
        else:
            raise ValueError(
                "No home_players or away_players, first load() the dataset"
            )

    def get_team_id_by_player_id(self, player_id):
        if hasattr(self, "home_players") and hasattr(self, "away_players"):
            for player in self.home_players + self.away_players:
                if player.id == player_id:
                    return player.team_id
        else:
            raise ValueError(
                "No home_players or away_players, first load() the dataset"
            )

    def sample(self, sample_rate: float):
        """Downsample the dataset by keeping every Nth frame.

        This method reduces the temporal resolution of the data by keeping only
        a subset of frames. Useful for faster experimentation or when full temporal
        resolution is not needed.

        Args:
            sample_rate: Sampling rate. For example:
                - 2.0 keeps every 2nd frame (halves data size)
                - 5.0 keeps every 5th frame (reduces to 20% of original)
                - 10.0 keeps every 10th frame (reduces to 10% of original)

        Returns:
            KloppyPolarsDataset: Self, for method chaining.

        Example:
            >>> # Keep every 2nd frame (50% of data)
            >>> dataset.sample(sample_rate=2.0)
            >>>
            >>> # Keep every 5th frame (20% of data)
            >>> dataset.sample(sample_rate=5.0)
            >>>
            >>> # Can chain with other methods
            >>> dataset.sample(5.0).add_dummy_labels().add_graph_ids()

        Note:
            This modifies ``self.data`` in-place. The original data is not preserved.

        Warning:
            Downsampling may affect velocity and acceleration calculations if you
            recalculate them after sampling. It's recommended to downsample before
            conversion to graphs.
        """
        sample = 1.0 / sample_rate

        self.data = self.data.filter((pl.col(Column.FRAME_ID) % sample) == 0)
        return self
