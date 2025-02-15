from .kloppy_polars import Column, Constant, Group

import polars as pl


def apply_speed_acceleration_filters(
    dataset: pl.DataFrame,
    max_ball_speed: float,
    max_player_speed: float,
    max_ball_acceleration: float,
    max_player_acceleration: float,
):
    return dataset.with_columns(
        pl.when(
            (pl.col(Column.OBJECT_ID) == Constant.BALL)
            & (pl.col(Column.SPEED) > max_ball_speed)
        )
        .then(max_ball_speed)
        .when(
            (pl.col(Column.OBJECT_ID) != Constant.BALL)
            & (pl.col(Column.SPEED) > max_player_speed)
        )
        .then(max_player_speed)
        .otherwise(pl.col(Column.SPEED))
        .alias(Column.SPEED)
    ).with_columns(
        pl.when(
            (pl.col(Column.OBJECT_ID) == Constant.BALL)
            & (pl.col(Column.ACCELERATION) > max_ball_acceleration)
        )
        .then(max_ball_acceleration)
        .when(
            (pl.col(Column.OBJECT_ID) != Constant.BALL)
            & (pl.col(Column.ACCELERATION) > max_player_acceleration)
        )
        .then(max_player_acceleration)
        .otherwise(pl.col(Column.ACCELERATION))
        .alias(Column.ACCELERATION)
    )
