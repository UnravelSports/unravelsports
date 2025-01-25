from kloppy.domain import (
    TrackingDataset,
    Frame,
    Orientation,
    DatasetTransformer,
    DatasetFlag,
    SecondSpectrumCoordinateSystem,
)

from typing import List, Dict, Union

from dataclasses import field, dataclass

from ...utils import DefaultDataset, add_dummy_label_column, add_graph_id_column

import polars as pl


DEFAULT_PLAYER_SMOOTHING_PARAMS = {"window_length": 7, "polyorder": 1}
DEFAULT_BALL_SMOOTHING_PARAMS = {"window_length": 3, "polyorder": 1}


@dataclass
class SoccerObject:
    id: Union[str, int]
    team_id: Union[str, int]
    position_name: str


@dataclass
class KloppyPolarsDataset(DefaultDataset):
    kloppy_dataset: TrackingDataset
    ball_carrier_threshold: float = None
    _identifier_column: str = field(default="id", init=False)
    _graph_id_column: str = field(default="graph_id")
    _label_column: str = field(default="label")
    _partition_by: List[str] = field(
        default_factory=lambda: ["id", "period_id"], init=False
    )
    _infer_ball_owning_team_id: bool = field(default=False, init=False)
    _overwrite_orientation: bool = field(default=False, init=False)
    _infer_goalkeepers: bool = field(default=False, init=False)
    
    def __transform_orientation(self):
        if not self.kloppy_dataset.metadata.flags & DatasetFlag.BALL_OWNING_TEAM:
            self._overwrite_orientation = True
            # In this package attacking is always left to right, so if this is not giving in Kloppy, overwrite it
            to_orientation = Orientation.STATIC_HOME_AWAY
        else:
            to_orientation = Orientation.BALL_OWNING_TEAM
            
        self.kloppy_dataset = DatasetTransformer.transform_dataset(
            dataset=self.kloppy_dataset,
            to_orientation=to_orientation,
            to_coordinate_system=SecondSpectrumCoordinateSystem(
                pitch_length=self.kloppy_dataset.metadata.pitch_dimensions.pitch_length,
                pitch_width=self.kloppy_dataset.metadata.pitch_dimensions.pitch_width,
            ),
        )
        return self.kloppy_dataset

    def __get_objects(self):
        def __artificial_game_id() -> str:
            from uuid import uuid4
            return str(uuid4())
        
        home_team, away_team = self.kloppy_dataset.metadata.teams
        
        if all(item is None for item in [p.starting_position for p in home_team.players]):
            self._infer_goalkeepers = True
            home_players = [
                SoccerObject(p.player_id, p.team.team_id, None)
                for p in home_team.players
            ]
            away_players = [
                SoccerObject(p.player_id, p.team.team_id, None)
                for p in away_team.players
            ]
        else:
            home_players = [
                SoccerObject(p.player_id, p.team.team_id, p.starting_position.code)
                for p in home_team.players
            ]
            away_players = [
                SoccerObject(p.player_id, p.team.team_id, p.starting_position.code)
                for p in away_team.players
            ]
        ball_object = SoccerObject("ball", "ball", "ball")
        game_id = self.kloppy_dataset.metadata.game_id
        if game_id is None:
            game_id = __artificial_game_id()
        return (home_players, away_players, ball_object, game_id)

    def __unpivot(self, object, coordinate):
        column = f"{object.id}_{coordinate}"

        return self.data.unpivot(
            index=[
                "period_id",
                "timestamp",
                "frame_id",
                "ball_state",
                "ball_owning_team_id",
            ],  # Columns to keep
            on=[column],
            value_name=coordinate,
            variable_name=self._identifier_column,
        ).with_columns(
            pl.col(self._identifier_column).str.replace(
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

        smoothed = df.group_by(self._partition_by, maintain_order=True).agg(
            [
                pl.col("vx")
                .map_elements(
                    lambda vx: savgol_filter(
                        vx,
                        window_length=smoothing_params["window_length"],
                        polyorder=smoothing_params["polyorder"],
                    ).tolist(),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias("vx_smoothed"),
                pl.col("vy")
                .map_elements(
                    lambda vy: savgol_filter(
                        vy,
                        window_length=smoothing_params["window_length"],
                        polyorder=smoothing_params["polyorder"],
                    ).tolist(),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias("vy_smoothed"),
                pl.col("vz")
                .map_elements(
                    lambda vy: savgol_filter(
                        vy,
                        window_length=smoothing_params["window_length"],
                        polyorder=smoothing_params["polyorder"],
                    ).tolist(),
                    return_dtype=pl.List(pl.Float64),
                )
                .alias("vz_smoothed"),
            ]
        )
        # Explode the smoothed columns back to original shape
        smoothed_exploded = smoothed.explode(
            ["vx_smoothed", "vy_smoothed", "vz_smoothed"]
        )
        # Combine with the original DataFrame if needed
        return df.with_columns(
            vx=smoothed_exploded["vx_smoothed"],
            vy=smoothed_exploded["vy_smoothed"],
            vz=smoothed_exploded["vz_smoothed"],
        )

    def __add_velocity(
        self,
        df: pl.DataFrame,
        player_smoothing_params: dict,
        ball_smoothing_params: dict,
    ):
        df = (
            df.sort(["id", "period_id", "timestamp", "team_id"], nulls_last=True)
            .with_columns(
                [
                    # Calculate differences within each group
                    pl.col("x").diff().over(self._partition_by).alias("dx"),
                    pl.col("y").diff().over(self._partition_by).alias("dy"),
                    pl.col("z").diff().over(self._partition_by).alias("dz"),
                    (pl.col("timestamp").dt.total_milliseconds() / 1_000)
                    .diff()
                    .over(self._partition_by)
                    .alias("dt"),
                ]
            )
            .with_columns(
                [
                    # Compute velocity components
                    (pl.col("dx") / pl.col("dt")).alias("vx"),
                    (pl.col("dy") / pl.col("dt")).alias("vy"),
                    (pl.col("dz") / pl.col("dt")).alias("vz"),
                ]
            )
            .with_columns(
                [
                    # Fill null values in vx and vy
                    pl.col("vx").fill_null(0).alias("vx"),
                    pl.col("vy").fill_null(0).alias("vy"),
                    pl.col("vz").fill_null(0).alias("vz"),
                ]
            )
        )

        if player_smoothing_params:
            player_df = self.__apply_smoothing(
                df=df.filter(pl.col(self._identifier_column) != self._ball_object.id),
                smoothing_params=player_smoothing_params,
            )
        else:
            player_df = df.filter(
                pl.col(self._identifier_column) != self._ball_object.id
            )

        if ball_smoothing_params:
            ball_df = self.__apply_smoothing(
                df.filter(pl.col(self._identifier_column) == self._ball_object.id),
                smoothing_params=ball_smoothing_params,
            )
        else:
            ball_df = df.filter(pl.col(self._identifier_column) == self._ball_object.id)
        df = pl.concat([player_df, ball_df])
        df = df.with_columns(
            [
                (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2)
                .sqrt()
                .alias("v")
            ]
        )

        return df

    def __add_acceleration(self, df: pl.DataFrame):
        df = (
            df.with_columns(
                [
                    # Calculate differences in vx, vy, and dt for acceleration
                    pl.col("vx").diff().over(self._partition_by).alias("dvx"),
                    pl.col("vy").diff().over(self._partition_by).alias("dvy"),
                    pl.col("vz").diff().over(self._partition_by).alias("dvz"),
                ]
            )
            .with_columns(
                [
                    # Compute ax and ay
                    (pl.col("dvx") / pl.col("dt")).alias("ax"),
                    (pl.col("dvy") / pl.col("dt")).alias("ay"),
                    (pl.col("dvz") / pl.col("dt")).alias("az"),
                ]
            )
            .with_columns(
                [
                    # Fill null values in vx and vy
                    pl.col("ax").fill_null(0).alias("ax"),
                    pl.col("ay").fill_null(0).alias("ay"),
                    pl.col("az").fill_null(0).alias("az"),
                ]
            )
            .with_columns(
                [
                    # Compute magnitude of acceleration a
                    (pl.col("ax") ** 2 + pl.col("ay") ** 2 + pl.col("az") ** 2)
                    .sqrt()
                    .alias("a")
                ]
            )
        )
        return df

    def __melt(
        self,
        home_players: List[SoccerObject],
        away_players: List[SoccerObject],
        ball_object: SoccerObject,
        game_id: Union[int, str],
    ):
        melted_dfs = []
        columns = self.data.columns

        for object in [ball_object] + home_players + away_players:
            melted_object_dfs = []
            for k, coordinate in enumerate(["x", "y", "z"]):
                if object.id != "ball" and coordinate == "z":
                    continue
                if not any(object.id in column for column in columns):
                    continue

                melted_df = self.__unpivot(object, coordinate)
                
                if object.id == "ball" and coordinate == "z":
                    if melted_df[coordinate].is_null().all():
                        melted_df = melted_df.with_columns([pl.lit(0.0).alias("z")])
                if k == 0:
                    melted_object_dfs.append(melted_df)
                else:
                    melted_object_dfs.append(melted_df[[coordinate]])

            if melted_object_dfs:
                object_df = pl.concat(melted_object_dfs, how="horizontal")
                if "z" not in object_df.columns:
                    object_df = object_df.with_columns([pl.lit(0.0).alias("z")])
                object_df = object_df.with_columns(
                    [
                        pl.lit(object.team_id).cast(pl.Utf8).alias("team_id"),
                        pl.lit(object.position_name).alias("position_name"),
                    ]
                )

                melted_dfs.append(object_df)
        
        df = pl.concat(melted_dfs, how="vertical")
        df = df.with_columns([pl.lit(game_id).alias("game_id")])
        df = df.sort(by=["period_id", "timestamp", "team_id"], nulls_last=True)
        return df
    
    def __get_inferred_ball_owning_team_id(self, df: pl.DataFrame):
        non_ball_owning_team = (
            df.filter(pl.col("ball_owning_team_id").is_null())
        )
        ball_owning_team = (
            df.filter(~pl.col("ball_owning_team_id").is_null())
        )
        
        ball = (
            non_ball_owning_team.filter(pl.col('team_id') == "ball")
        )
        players = (
            non_ball_owning_team.filter(pl.col('team_id') != "ball")
        )
        result = (
            players.drop('ball_owning_team_id')
            .join(
                ball.select(
                    ['game_id', 'period_id', 'frame_id', 
                    pl.col('x').alias('ball_x'),
                    pl.col('y').alias('ball_y'), 
                    pl.col('z').alias('ball_z')]
                ),
                on=['game_id', 'period_id', 'frame_id'],
                how='left'
            )
            .with_columns([
                ((pl.col('x') - pl.col('ball_x'))**2 + 
                (pl.col('y') - pl.col('ball_y'))**2 + 
                (pl.col('z') - pl.col('ball_z'))**2
                ).sqrt().alias('distance')
            ])
            .group_by(['game_id', 'period_id', 'frame_id'])
            .agg([
                pl.when(pl.col('distance').min() < self.ball_carrier_threshold)
                .then(pl.col('team_id').filter(pl.col('distance') == pl.col('distance').min()).first())
                .otherwise(None)
                .alias('ball_owning_team_id'),
                pl.all().sort_by('distance').first()
            ])
        )
        non_ball_owning_team = (
            non_ball_owning_team.drop('ball_owning_team_id')
            .join(
                result.select(['game_id', 'period_id', 'frame_id', 'ball_owning_team_id']),
                on=['game_id', 'period_id', 'frame_id'],
                how='left'
            )
            .filter(
                ~pl.col("ball_owning_team_id").is_null()
            )
            .with_columns([
                pl.col("ball_owning_team_id").cast(ball_owning_team.schema['team_id'])
            ])
            .select(ball_owning_team.columns)
        )
        ball_owning_team = (
            ball_owning_team
            .with_columns([
                pl.col("ball_owning_team_id").cast(ball_owning_team.schema['team_id'])
            ])
        )
        
        new_df = (
            pl.concat([
                ball_owning_team,
                non_ball_owning_team
            ], how="vertical")
            .sort(['game_id', 'period_id', 'frame_id', 'team_id'])
        )
        return new_df
    
    def __get_inferred_goalkeepers(self, df: pl.DataFrame):
        goal_x = self.pitch_dimensions.pitch_length / 2
        goal_y = 0
        
        df_with_distances = (
            df.filter(pl.col('team_id') != "ball")
            .with_columns([
                ((pl.col('x') - (-goal_x))**2 + (pl.col('y') - goal_y)**2).sqrt().alias('dist_left'),
                ((pl.col('x') - goal_x)**2 + (pl.col('y') - goal_y)**2).sqrt().alias('dist_right')
            ])
        )
        result = (
            df_with_distances
            .with_columns([
                pl.col('dist_left').min().over(['game_id', 'period_id', 'frame_id', 'team_id']).alias('min_dist_left'),
                pl.col('dist_right').min().over(['game_id', 'period_id', 'frame_id', 'team_id']).alias('min_dist_right')
            ])
            .with_columns([
                pl.when(pl.col('team_id') == pl.col('ball_owning_team_id'))
                .then(
                    pl.when(pl.col('dist_left') == pl.col('min_dist_left'))
                    .then(pl.lit('GK'))
                    .otherwise(None)
                )
                .otherwise(
                    pl.when(pl.col('dist_right') == pl.col('min_dist_right'))
                    .then(pl.lit('GK'))
                    .otherwise(None)
                )
                .alias('position_name')
            ])
            .drop(['min_dist_left', 'min_dist_right', 'dist_left', 'dist_right'])
        )
        ball_rows = df.filter(pl.col('team_id') == "ball")
        non_ball_rows = result

        return (
            pl.concat([ball_rows, non_ball_rows], how="vertical")
            .sort(['game_id', 'period_id', 'frame_id', 'team_id'])
        )
        
    def __fix_orientation_to_ball_owning(self, df: pl.DataFrame, home_team_id: Union[str, int]):
        # When _overwrite_orientation is True, it means the orientation is "STATIC_HOME_AWAY"
        # This means that when away is the attacking team we can flip all coordinates by -1.0
        
        flip_columns = ['x', 'y', 'vx', 'vy', 'ax', 'ay']
        
        return df.with_columns([
            pl.when(pl.col('ball_owning_team_id').cast(str) != str(home_team_id))
            .then(pl.col(flip_columns) * -1)
            .otherwise(pl.col(flip_columns))
        ])

    def load(
        self,
        player_smoothing_params: Union[dict, None] = DEFAULT_PLAYER_SMOOTHING_PARAMS,
        ball_smoothing_params: Union[dict, None] = DEFAULT_BALL_SMOOTHING_PARAMS,
    ):
        if self.kloppy_dataset.metadata.orientation == Orientation.NOT_SET:
            raise ValueError("Data sources with an undefined orientation can not be used inside the 'unravelsports' package...")
        
        self.kloppy_dataset = self.__transform_orientation()
        self.pitch_dimensions = self.kloppy_dataset.metadata.pitch_dimensions

        self.data = self.kloppy_dataset.to_df(engine="polars")
        (self._home_players, self._away_players, self._ball_object, self._game_id) = (
            self.__get_objects()
        )
        df = self.__melt(
            self._home_players, self._away_players, self._ball_object, self._game_id
        )
        
        df = self.__add_velocity(df, player_smoothing_params, ball_smoothing_params)
        df = self.__add_acceleration(df)
        df = df.drop(["dx", "dy", "dz", "dt", "dvx", "dvy", "dvz"])
        
        df = df.filter(
            ~(pl.col('x').is_null() & pl.col('y').is_null())
        )
        
        if df['ball_owning_team_id'].is_null().all() and self.ball_carrier_threshold:
                raise ValueError("This dataset requires us to infer the ball_owning_team_id, please specifiy a ball_carrier_threshold (float) to do so.")
        
        if self.ball_carrier_threshold is not None:
            df = self.__get_inferred_ball_owning_team_id(df)
            
        if self._overwrite_orientation:
            home_team, _ = self.kloppy_dataset.metadata.teams
            df = self.__fix_orientation_to_ball_owning(df, home_team_id=home_team.team_id)
        
        if self._infer_goalkeepers:
            df = self.__get_inferred_goalkeepers(df)
        
        self.data = df
        return self.data, self.pitch_dimensions

    def add_dummy_labels(
        self,
        by: List[str] = ["game_id", "frame_id"]
    ) -> pl.DataFrame:
        self.data = add_dummy_label_column(self.data, by, self._label_column)
        return self.data

    def add_graph_ids(
        self, by: List[str] = ["game_id", "period_id"]
    ) -> pl.DataFrame:
        self.data = add_graph_id_column(self.data, by, self._graph_id_column)
        return self.data
