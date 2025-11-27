from dataclasses import dataclass

import numpy as np

from typing import Literal, List, Union, Literal, Optional

from kloppy.domain import AttackingDirection, Orientation

from .detection import FormationDetection, Formations

import polars as pl

from ...dataset.kloppy_polars import (
    Group,
    Column,
    Constant,
)


@dataclass
class EFPI(FormationDetection):
    _fit = False

    def __post_init__(self):
        super().__post_init__()
        self.__get_linear_sum_assignment()

    def __get_linear_sum_assignment(self):
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            raise ImportError(
                "Seems like you don't have scipy installed. Please"
                " install it using: pip install scipy"
            )
        self.linear_sum_assignment = linear_sum_assignment

    def __repr__(self):
        if not self._fit:
            return f"EFPI(n_frames={len(self.dataset)}, formations={self.formations if self._formations is not None else 'mplsoccer'})"
        else:
            return f"EFPI(n_frames={len(self.dataset)}, formations={self.formations if self._formations is not None else 'mplsoccer'}, every={self._every}, substitutions={self._substitutions}, change_after_possession={self._change_after_possession}, change_threshold={self._change_threshold})"

    @staticmethod
    def __scale_all_to_bounds(points, min_x, min_y, max_x, max_y):
        global_min = points.min(axis=(0, 1))
        global_max = points.max(axis=(0, 1))

        scale = np.where(
            global_max - global_min != 0,
            (max_x - min_x, max_y - min_y) / (global_max - global_min),
            1,
        )

        # Apply transformation
        scaled_points = (points - global_min) * scale + np.array([min_x, min_y])

        return scaled_points

    def __assign_formation(
        self, coordinates: np.ndarray, direction: AttackingDirection
    ):
        if direction == AttackingDirection.LTR:
            relevant_formations = self._forms.get_formation_positions_left_to_right()
            relevant_position_labels = self._forms.get_formation_labels_left_to_right()
        elif direction == AttackingDirection.RTL:
            relevant_formations = self._forms.get_formation_positions_right_to_left()
            relevant_position_labels = self._forms.get_formation_labels_right_to_left()
        else:
            raise ValueError("AttackingDirection is not set...")

        numb_players = len(coordinates)

        min_x, max_x = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
        min_y, max_y = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])

        _form = np.asarray([v for k, v in relevant_formations[numb_players].items()])
        _form = self.__scale_all_to_bounds(
            points=_form, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y
        )
        forms = [k for k in relevant_formations[numb_players]]

        cost_matrices = np.linalg.norm(
            coordinates[:, np.newaxis, np.newaxis, :] - _form[np.newaxis, :, :, :],
            axis=-1,
        )

        costs = np.array(
            [
                cost_matrices[:, i, :][
                    self.linear_sum_assignment(cost_matrices[:, i, :])
                ].sum()
                for i in range(len(_form))
            ]
        )

        idx = np.argmin(costs)
        selected_formation_cost = np.min(costs)

        cheapest_matrix = cost_matrices[:, idx, :]
        row_ind, col_ind = self.linear_sum_assignment(cheapest_matrix)

        selected_formation = forms[idx]
        selected_coords = relevant_formations[numb_players][selected_formation]
        players = relevant_position_labels[numb_players][selected_formation][row_ind][
            col_ind
        ]
        return (
            players,
            selected_coords,
            coordinates,
            selected_formation,
            selected_formation_cost,
        )

    def __is_update(self, team_id, formation_cost, object_ids, is_attack):
        if self._forms.detected_formations.get(team_id) is None:
            return True
        else:
            if self._change_threshold is None:
                return True
            elif set(self._forms.detected_formations[team_id].ids) != set(object_ids):
                # update if we encounter a different set of player ids frame to frame
                return True
            elif (self._change_after_possession) & (
                self._forms.detected_formations[team_id].is_attack != is_attack
            ):
                # update if we switch from attack to defense
                return True
            elif (
                self._forms.detected_formations[team_id].cost - formation_cost
            ) / formation_cost > self._change_threshold:
                # update if we passed the threshold
                return True
            else:
                return False

    def __detect(self, is_attack, direction, d):
        xs, ys = d[Column.X], d[Column.Y]
        if is_attack:
            team_idx = np.where(
                (d[Column.TEAM_ID] == d[Column.BALL_OWNING_TEAM_ID])
                & (d[Column.POSITION_NAME] != "GK")
            )[0]
            gk_idx = np.where(
                (d[Column.TEAM_ID] == d[Column.BALL_OWNING_TEAM_ID])
                & (d[Column.POSITION_NAME] == "GK")
            )[0]
            team_id = d[Column.BALL_OWNING_TEAM_ID][0]
        else:
            team_idx = np.where(
                (d[Column.TEAM_ID] != d[Column.BALL_OWNING_TEAM_ID])
                & (d[Column.POSITION_NAME] != "GK")
                & (d[Column.TEAM_ID] != Constant.BALL)
            )[0]
            gk_idx = np.where(
                (d[Column.TEAM_ID] != d[Column.BALL_OWNING_TEAM_ID])
                & (d[Column.TEAM_ID] != Constant.BALL)
                & (d[Column.POSITION_NAME] == "GK")
            )[0]
            team_id = d[Column.TEAM_ID][
                (d[Column.TEAM_ID] != d[Column.BALL_OWNING_TEAM_ID])
                & (d[Column.TEAM_ID] != Constant.BALL)
            ][0]

        outfield_coordinates = np.stack((xs[team_idx], ys[team_idx]), axis=-1)

        position_labels, _, _, formation, formation_cost = self.__assign_formation(
            coordinates=outfield_coordinates, direction=direction
        )

        _idxs = np.concatenate((team_idx, gk_idx))
        labels = np.concatenate((position_labels, ["GK"]))
        object_ids = d[Column.OBJECT_ID][_idxs]

        if self.__is_update(team_id, formation_cost, object_ids, is_attack):
            self._forms.set_detected_formation(
                team_id=team_id,
                is_attack=is_attack,
                name=formation,
                cost=formation_cost,
                labels=labels,
                ids=object_ids,
            )

    def _compute(self, args: List[pl.Series], **kwargs) -> pl.DataFrame:
        d = {col: args[i].to_numpy() for i, col in enumerate(self._exprs_variables)}

        d.update(kwargs)

        attacking_team_id = d[Column.BALL_OWNING_TEAM_ID][0]
        attacking_direction = (
            AttackingDirection.LTR
            if self.settings.orientation == Orientation.BALL_OWNING_TEAM
            else (
                AttackingDirection.LTR
                if self.settings.orientation == Orientation.STATIC_HOME_AWAY
                and attacking_team_id == self.settings.home_team_id
                else AttackingDirection.RTL
            )
        )
        defending_direction = (
            AttackingDirection.RTL
            if attacking_direction == AttackingDirection.LTR
            else AttackingDirection.LTR
        )

        self.__detect(
            is_attack=True,
            direction=attacking_direction,
            d=d,
        )
        self.__detect(
            is_attack=False,
            direction=defending_direction,
            d=d,
        )

        return self._forms.get_detected_formations_as_dict(
            object_ids=d[Column.OBJECT_ID].tolist(), team_ids=d[Column.TEAM_ID].tolist()
        )

    @property
    def return_dtypes(self):
        return pl.Struct({
            Column.OBJECT_ID: pl.List(pl.String),  
            Column.TEAM_ID: pl.List(pl.String),    
            "position": pl.List(pl.String),
            "formation": pl.List(pl.String),
        })
        
    def fit(
        self,
        start_time: pl.duration = None,
        end_time: pl.duration = None,
        period_id: int = None,
        every: Optional[
            Union[str, Literal["frame"], Literal["period"], Literal["possession"]]
        ] = "frame",
        formations: Union[List[str], Literal["shaw-glickman"]] = None,
        substitutions: Literal["merge", "drop"] = "drop",
        change_after_possession: bool = True,
        change_threshold: float = None,
    ):
        """
        - Count number of players seen
        - update_threshold: float: value between 0 and 1 indicating the minimum change in formation assignment cost to update the detected formation.
        """
        self._substitutions = substitutions
        self._change_threshold = change_threshold
        self._change_after_possession = change_after_possession
        self._every = every
        self._formations = formations

        __added_arbitrary_base = False

        self._forms = Formations(
            pitch_length=self.settings.pitch_dimensions.pitch_length,
            pitch_width=self.settings.pitch_dimensions.pitch_width,
            formations=self._formations,
        )

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

        if self._every == "frame":
            group_by_columns = Group.BY_FRAME

            self.output = (
                (
                    df.sort([Column.FRAME_ID, Column.OBJECT_ID])
                    .group_by(group_by_columns, maintain_order=True)
                    .agg(
                        pl.map_groups(
                            exprs=self._exprs_variables,
                            function=lambda group: self._compute(group),
                            return_dtype=self.return_dtypes,
                            returns_scalar=True
                        ).alias("result")
                    )
                    .unnest("result")
                )
                .explode([Column.OBJECT_ID, Column.TEAM_ID, "position", "formation"])
                .join(
                    df.select([Column.FRAME_ID, Column.BALL_OWNING_TEAM_ID]).unique(
                        [Column.FRAME_ID, Column.BALL_OWNING_TEAM_ID]
                    ),
                    on=Column.FRAME_ID,
                    how="left",
                )
                .with_columns(
                    pl.when((pl.col(Column.OBJECT_ID) == Constant.BALL))
                    .then(None)
                    .when(
                        (pl.col(Column.TEAM_ID) == pl.col(Column.BALL_OWNING_TEAM_ID))
                    )
                    .then(True)
                    .otherwise(False)
                    .alias("is_attacking")
                )
                .sort([Column.FRAME_ID, "is_attacking", Column.OBJECT_ID])
            )
            self.segments = None
            self._fit = True
            return self

        elif isinstance(self._every, str):
            group_by_columns = [
                Column.GAME_ID,
                Column.PERIOD_ID,
                Column.BALL_OWNING_TEAM_ID,
                Column.OBJECT_ID,
            ]
            segment_id = f"{self._every}_id"

            df = df.with_columns(
                [
                    (
                        pl.col(Column.BALL_OWNING_TEAM_ID) == pl.col(Column.TEAM_ID)
                    ).alias("is_attacking")
                ]
            )
            group_by_columns.append("is_attacking")

            if self._every == "possession":
                df1 = df.sort(Column.FRAME_ID).with_columns(
                    [
                        (
                            (
                                pl.col(Column.BALL_OWNING_TEAM_ID)
                                != pl.col(Column.BALL_OWNING_TEAM_ID).shift(1)
                            )
                            | (
                                pl.col(Column.PERIOD_ID)
                                != pl.col(Column.PERIOD_ID).shift(1)
                            )
                        )
                        .fill_null(True)
                        .cast(pl.Int32)
                        .cum_sum()
                        .alias(segment_id)
                    ]
                )
            elif self._every == "period":
                df1 = df.sort("frame_id")

            elif isinstance(self._every, str):
                from datetime import datetime

                base_time = datetime(2000, 1, 1)
                __added_arbitrary_base = True

                df1 = df.sort(Column.FRAME_ID).with_columns(
                    (pl.lit(base_time) + pl.col(Column.TIMESTAMP))
                    .dt.truncate(self._every)
                    .alias(segment_id)
                )

            # Any moment we have more than 11 players we have overlapping substitutions in a segment
            overlapping_substitutions = (
                df1.filter(
                    (pl.col(Column.TEAM_ID) != Constant.BALL)
                    & (pl.col(Column.POSITION_NAME) != "GK")
                )
                .group_by(
                    [Column.GAME_ID, Column.PERIOD_ID, Column.TEAM_ID, segment_id]
                    if self._every != "period"
                    else [Column.GAME_ID, Column.PERIOD_ID, Column.TEAM_ID], maintain_order=True
                )
                .agg([pl.col(Column.OBJECT_ID).n_unique().alias("objects")])
                .sort([segment_id])
                .filter(pl.col("objects") > 10)
            )

            if not overlapping_substitutions.is_empty():
                if self._substitutions == "drop":
                    columns = [
                        Column.GAME_ID,
                        Column.PERIOD_ID,
                        Column.TEAM_ID,
                        Column.OBJECT_ID,
                        segment_id,
                    ]
                    player_segments_to_drop = (
                        df1.join(
                            overlapping_substitutions,
                            how="inner",
                            on=[
                                Column.GAME_ID,
                                Column.PERIOD_ID,
                                Column.TEAM_ID,
                                segment_id,
                            ],
                        )
                        .group_by(columns, maintain_order=True)
                        .agg([pl.len().alias("length")])
                        .with_columns(
                            pl.col("length")
                            .rank(method="ordinal", descending=True)
                            .over(
                                [
                                    Column.GAME_ID,
                                    Column.PERIOD_ID,
                                    Column.TEAM_ID,
                                    segment_id,
                                ]
                            )
                            .alias("rank")
                        )
                        .filter(pl.col("rank") > 11)
                        .drop("rank")
                        .select(columns)
                    )
                    df1 = df1.join(player_segments_to_drop, on=columns, how="anti")
                elif self._substitutions == "merge":
                    raise NotImplementedError(
                        "Merging overlapping substitutions within a window is not implemented yet..."
                    )
                else:
                    raise ValueError(
                        "'substitutions' should either be 'merge' or 'drop'..."
                    )

            segment_coordinates = (
                df1.group_by(
                    group_by_columns + [segment_id]
                    if self._every != "period"
                    else group_by_columns, maintain_order=True
                )
                .agg(
                    [
                        pl.col(Column.X).mean().alias(Column.X),
                        pl.col(Column.Y).mean().alias(Column.Y),
                        pl.col(Column.POSITION_NAME)
                        .first()
                        .alias(Column.POSITION_NAME),
                        pl.col(Column.TEAM_ID).first().alias(Column.TEAM_ID),
                        pl.col(Column.FRAME_ID).unique().len().alias("n_frames"),
                        pl.col(Column.TIMESTAMP).min().alias("start_timestamp"),
                        pl.col(Column.TIMESTAMP).max().alias("end_timestamp"),
                        pl.col(Column.FRAME_ID).min().alias("start_frame_id"),
                        pl.col(Column.FRAME_ID).max().alias("end_frame_id"),
                    ]
                )
                .sort([Column.PERIOD_ID, segment_id, Column.OBJECT_ID])
            )

            positions = (
                (
                    segment_coordinates.group_by(
                        (
                            [
                                Column.GAME_ID,
                                Column.PERIOD_ID,
                                Column.BALL_OWNING_TEAM_ID,
                            ]
                            + [segment_id]
                            if self._every != "period"
                            else [
                                Column.GAME_ID,
                                Column.PERIOD_ID,
                                Column.BALL_OWNING_TEAM_ID,
                            ]
                        ),
                        maintain_order=True,
                    )
                    .agg(
                        pl.map_groups(
                            exprs=self._exprs_variables,
                            function=lambda group: self._compute(group),
                            return_dtype=self.return_dtypes,
                            returns_scalar=True
                        ).alias("result")
                    )
                    .unnest("result")
                )
                .explode([Column.OBJECT_ID, Column.TEAM_ID, "position", "formation"])
                .with_columns(
                    pl.when((pl.col(Column.OBJECT_ID) == Constant.BALL))
                    .then(None)
                    .when(
                        (pl.col(Column.TEAM_ID) == pl.col(Column.BALL_OWNING_TEAM_ID))
                    )
                    .then(True)
                    .otherwise(False)
                    .alias("is_attacking")
                )
            )

        if __added_arbitrary_base:
            positions = positions.with_columns(
                (pl.col(segment_id) - pl.lit(base_time))
                .cast(pl.Duration)
                .alias(segment_id)
            )

        self.output = positions.sort([segment_id, "is_attacking", Column.OBJECT_ID])

        self.segments = (
            segment_coordinates.select(
                [
                    segment_id,
                    "n_frames",
                    "start_timestamp",
                    "end_timestamp",
                    "start_frame_id",
                    "end_frame_id",
                ]
            )
            .unique()
            .sort([segment_id])
        )
        self._fit = True
        return self
