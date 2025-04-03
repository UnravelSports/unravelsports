import numpy as np
import polars as pl

from dataclasses import dataclass, field

from typing import Literal, List, Union

from ..dataset.kloppy_polars import (
    KloppyPolarsDataset,
    MetricPitchDimensions,
    Group,
    Column,
    Constant,
)

from .utils import time_to_intercept, probability_to_intercept


@dataclass
class PressingIntensity:
    dataset: KloppyPolarsDataset
    chunk_size: int = field(init=True, repr=False, default=2_0000)

    _method: str = field(init=False, repr=False, default="teams")
    _ball_method: str = field(init=False, repr=False, default="max")
    _speed_threshold: float = field(init=False, repr=False, default=None)
    _reaction_time: float = field(init=False, repr=False, default=0.7)
    _sigma: float = field(init=False, repr=False, default=0.45)
    _time_threshold: float = field(init=False, repr=False, default=1.5)
    _orient: str = field(init=False, repr=False, default="ball_owning")
    _line_method: str = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if not isinstance(self.dataset, KloppyPolarsDataset):
            raise ValueError("dataset should be of type KloppyPolarsDataset...")

        self.settings = self.dataset.settings
        self.dataset = self.dataset.data

    def __repr__(self):
        n_frames = (
            self.output[Column.FRAME_ID].n_unique() if hasattr(self, "output") else None
        )
        return f"PressingIntensity(n_frames={n_frames})"

    @property
    def __exprs_variables(self):
        return [
            Column.X,
            Column.Y,
            Column.Z,
            Column.VX,
            Column.VY,
            Column.VZ,
            Column.SPEED,
            Column.TEAM_ID,
            Column.BALL_OWNING_TEAM_ID,
            Column.OBJECT_ID,
            Column.IS_BALL_CARRIER,
        ]

    def __compute(self, args: List[pl.Series]) -> dict:
        def _set_minimum(matrix, ball_carrier_idx, ball_idx):
            # Take the element-wise maximum of the ball carrier and the ball
            matrix[:, ball_carrier_idx] = np.minimum(
                matrix[:, ball_carrier_idx], matrix[:, ball_idx]
            )
            # Delete ball column
            matrix = np.delete(matrix, ball_idx, axis=1)
            return matrix

        d = {col: args[i].to_numpy() for i, col in enumerate(self.__exprs_variables)}

        ball_idx, ball_carrier_idx = None, None

        if self._ball_method in ["max", "include"]:
            ball_mask = d[Column.TEAM_ID] == Constant.BALL
            ball_owning_mask = (d[Column.TEAM_ID] == d[Column.BALL_OWNING_TEAM_ID]) | (
                ball_mask
            )
            non_ball_owning_mask = ~ball_owning_mask

        elif self._ball_method == "exclude":
            ball_mask = d[Column.TEAM_ID] != Constant.BALL
            ball_owning_mask = (d[Column.TEAM_ID] == d[Column.BALL_OWNING_TEAM_ID]) & (
                ball_mask
            )
            non_ball_owning_mask = (
                d[Column.TEAM_ID] != d[Column.BALL_OWNING_TEAM_ID]
            ) & ball_mask

        if self._method == "teams":
            ball_owning_idxs = np.where(ball_owning_mask)[0]
            non_ball_owning_idxs = np.where(non_ball_owning_mask)[0]

            if self._ball_method == "max":
                ball_idx = np.where(
                    d[Column.TEAM_ID][ball_owning_idxs] == Constant.BALL
                )[0][0]
                ball_carrier_idx = np.where(
                    d[Column.IS_BALL_CARRIER][ball_owning_idxs]
                )[0][0]

            xs1, ys1, zs1 = (
                d[Column.X][ball_owning_idxs],
                d[Column.Y][ball_owning_idxs],
                d[Column.Z][ball_owning_idxs],
            )
            xs2, ys2, zs2 = (
                d[Column.X][non_ball_owning_idxs],
                d[Column.Y][non_ball_owning_idxs],
                d[Column.Z][non_ball_owning_idxs],
            )

            vxs1, vys1, vzs1 = (
                d[Column.VX][ball_owning_idxs],
                d[Column.VY][ball_owning_idxs],
                d[Column.VZ][ball_owning_idxs],
            )
            vxs2, vys2, vzs2 = (
                d[Column.VX][non_ball_owning_idxs],
                d[Column.VY][non_ball_owning_idxs],
                d[Column.VZ][non_ball_owning_idxs],
            )
            column_objects, row_objects = (
                d[Column.OBJECT_ID][ball_owning_idxs],
                d[Column.OBJECT_ID][non_ball_owning_idxs],
            )

            if self._speed_threshold:
                column_mask = d[Column.SPEED][ball_owning_idxs] < self._speed_threshold
                row_mask = d[Column.SPEED][non_ball_owning_idxs] < self._speed_threshold

        elif self._method == "full":
            if self._ball_method == "exclude":
                mask = np.where(ball_mask)[0]
            else:
                mask = np.where(d[Column.TEAM_ID] == d[Column.TEAM_ID])[0]

            if self._ball_method == "max":
                ball_idx = np.where(ball_mask)[0][0]
                ball_carrier_idx = np.where(d[Column.IS_BALL_CARRIER][mask])[0][0]

            xs1, ys1, zs1 = xs2, ys2, zs2 = (
                d[Column.X][mask],
                d[Column.Y][mask],
                d[Column.Z][mask],
            )
            vxs1, vys1, vzs1 = vxs2, vys2, vzs2 = (
                d[Column.VX][mask],
                d[Column.VY][mask],
                d[Column.VZ][mask],
            )
            column_objects, row_objects = (
                d[Column.OBJECT_ID][mask],
                d[Column.OBJECT_ID][mask],
            )

            if self._speed_threshold:
                column_mask = d[Column.SPEED][mask] < self._speed_threshold
                row_mask = d[Column.SPEED][mask] < self._speed_threshold

        if ball_idx is not None:
            column_objects = np.delete(column_objects, ball_idx, axis=0)
            if self._speed_threshold:
                column_mask = np.delete(column_mask, ball_idx, axis=0)
        
        print(list(xs1))
        print(list(ys1))
        print(list(vxs1))
        print(list(vys1))
        print(self.settings.pitch_dimensions)
        print('---')

        if self._line_method is not None:
            if self._line_method == "touchline":
                pass
            elif self._line_method == "byline":
                pass
            elif self._line_method == "all":
                pass

        p1 = np.stack((xs1, ys1, zs1), axis=-1)
        p2 = np.stack((xs2, ys2, zs2), axis=-1)
        v1 = np.stack((vxs1, vys1, vzs1), axis=-1)
        v2 = np.stack((vxs2, vys2, vzs2), axis=-1)

        tti = time_to_intercept(
            p1=p1,
            p2=p2,
            v1=v1,
            v2=v2,
            reaction_time=self._reaction_time,
            max_object_speed=self.settings.max_player_speed,
        )
        if self._ball_method == "max":
            tti = _set_minimum(
                matrix=tti, ball_carrier_idx=ball_carrier_idx, ball_idx=ball_idx
            )
            if self._method == "full":
                tti = np.delete(tti, ball_idx, axis=0)
                row_objects = np.delete(row_objects, ball_idx, axis=0)
                if self._speed_threshold:
                    row_mask = np.delete(row_mask, ball_idx, axis=0)

        pti = probability_to_intercept(
            time_to_intercept=tti,
            tti_sigma=self._sigma,
            tti_time_threshold=self._time_threshold,
        )

        if self._method == "full":
            np.fill_diagonal(tti, np.inf)
            np.fill_diagonal(tti, 0.0)

        if self._speed_threshold:
            pti[row_mask, :] = 0.0
            pti[:, column_mask] = 0.0

        if (
            (
                (self._orient == "away_home")
                & (d[Column.BALL_OWNING_TEAM_ID][0] != self.settings.home_team_id)
            )
            | (
                (self._orient == "home_away")
                & (d[Column.BALL_OWNING_TEAM_ID][0] == self.settings.home_team_id)
            )
            | (self._orient == "pressing")
        ):
            return {
                "time_to_intercept": tti.T.tolist(),
                "probability_to_intercept": pti.T.tolist(),
                "columns": row_objects.tolist(),
                "rows": column_objects.tolist(),
            }

        return {
            "time_to_intercept": tti.tolist(),
            "probability_to_intercept": pti.tolist(),
            "columns": column_objects.tolist(),
            "rows": row_objects.tolist(),
        }

    def fit(
        self,
        start_time: pl.duration = None,
        end_time: pl.duration = None,
        period_id: int = None,
        speed_threshold: float = None,
        reaction_time: float = 0.7,
        time_threshold: float = 1.5,
        sigma: float = 0.45,
        method: Literal["teams", "full"] = "teams",
        ball_method: Literal["include", "exclude", "max"] = "max",
        orient: Literal[
            "ball_owning", "pressing", "home_away", "away_home"
        ] = "ball_owning",
        line_method: Union[None, Literal["touchline", "byline", "all"]] = None
    ):
        """
        method: str ["teams", "full"]
            "teams" creates a 11x11 matrix, "full" creates a 22x22 matrix
        ball_method: str ["include", "exclude", "max"]
            "include" creates a 11x12 matrix
            "exclude" ignores ball
            "max" keeps 11x11 but ball carrier pressing intensity is now max(ball, ball_carrier)
        speed_threshold: float.
            Masks pressing intensity to only include players travelling above a certain speed
            threshold in meters per second.
        orient: str ["ball_owning", "pressing", "home_away", "away_home"]
            Pressing Intensity output as seen from the 'row' perspective.
            method and orient are in sync, meaning "full" and "away_home" sorts row and columns
            such that the away team players are displayed first
        """
        if period_id is not None and not isinstance(period_id, int):
            raise TypeError("period_id should be of type integer")
        if method not in ["teams", "full"]:
            raise ValueError("method should be 'teams' or 'full'")
        if ball_method not in ["include", "exclude", "max"]:
            raise ValueError("ball_method should be 'include', 'exclude' or 'max'")
        if orient not in ["ball_owning", "pressing", "home_away", "away_home"]:
            raise ValueError(
                "orient should be 'ball_owning', 'pressing', 'home_away', 'away_home'"
            )
        if line_method is not None and line_method not in ["touchline", "byline", "all"]:
            raise ValueError(
                "line_method should be 'touchline', 'byline', 'all' or None"
            )
        if not isinstance(reaction_time, Union[float, int]):
            raise TypeError("reaction_time should be of type float")
        if speed_threshold is not None and not isinstance(
            speed_threshold, Union[float, int]
        ):
            raise TypeError("speed_threshold should be of type float (or None)")
        if not isinstance(time_threshold, Union[float, int]):
            raise TypeError("time_threshold should be of type float")
        if not isinstance(sigma, Union[float, int]):
            raise TypeError("sigma should be of type float")

        self._method = method
        self._ball_method = ball_method
        self._speed_threshold = speed_threshold
        self._reaction_time = reaction_time
        self._time_threshold = time_threshold
        self._sigma = sigma
        self._orient = orient
        self._line_method = line_method

        if all(x is None for x in [start_time, end_time, period_id]):
            df = self.dataset
        elif all(x is not None for x in [start_time, end_time, period_id]):
            df = self.dataset.filter(
                (pl.col(Column.TIMESTAMP).is_between(start_time, end_time))
                & (pl.col(Column.PERIOD_ID) == period_id)
            )
        else:
            raise ValueError(
                "Please specificy all of start_time, end_time and period_id or none of them..."
            )

        sort_descending = [False] * len(Group.BY_TIMESTAMP)
        if self._orient in ["home_away", "away_home"]:
            alias = "is_home"
            sort_by = Group.BY_TIMESTAMP + [alias]
            sort_descending = sort_descending + (
                [True] if self._orient == "home_away" else [False]
            )
            with_columns = [
                pl.when(pl.col(Column.TEAM_ID) == self.settings.home_team_id)
                .then(True)
                .when(pl.col(Column.TEAM_ID) == Constant.BALL)
                .then(None)
                .otherwise(False)
                .alias(alias)
            ]
        elif self._orient in ["ball_owning", "pressing"]:
            alias = "is_ball_owning"
            sort_by = Group.BY_TIMESTAMP + [alias]
            sort_descending = sort_descending + (
                [True] if self._orient == "ball_owning" else [False]
            )
            with_columns = [
                pl.when(pl.col(Column.TEAM_ID) == pl.col(Column.BALL_OWNING_TEAM_ID))
                .then(True)
                .when(pl.col(Column.TEAM_ID) == Constant.BALL)
                .then(None)
                .otherwise(False)
                .alias(alias)
            ]

        self.output = (
            df.with_columns(with_columns)
            .sort(by=sort_by, descending=sort_descending, nulls_last=True)
            .group_by(Group.BY_TIMESTAMP, maintain_order=True)
            .agg(
                pl.map_groups(
                    exprs=self.__exprs_variables,
                    function=self.__compute,
                ).alias("results")
            )
            .unnest("results")
        )

        return self
