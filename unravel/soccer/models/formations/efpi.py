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
    """Detect soccer team formations using Expected Formation Positioning Inference (EFPI).

    EFPI automatically identifies team formations (e.g., 4-3-3, 4-4-2, 3-5-2) from player
    positions using optimal assignment between observed positions and canonical formation
    templates. The algorithm uses the Hungarian algorithm (linear sum assignment) to
    minimize the total distance between players and template positions, scaled to match
    the team's spatial distribution.

    The method works by:
    1. Extracting player positions for each team (attack/defense separately)
    2. Comparing positions to predefined formation templates (mplsoccer or Shaw-Glickman)
    3. Finding the best-fit formation via optimal bipartite matching
    4. Assigning positional labels (e.g., "LW", "CM", "RB") to each player
    5. Tracking formation changes over time or possession segments

    Key features:
    - Automatic formation detection with no manual labeling
    - Separate formations for attacking and defending phases
    - Position-specific labels for each player
    - Temporal aggregation (per-frame, per-possession, or custom windows)
    - Substitution handling (merge or drop)
    - Formation stability tracking via cost thresholds

    The algorithm is based on research in formation detection and extends methods from
    Decroos et al. and Shaw & Glickman's formation analysis work.

    Args:
        dataset (KloppyPolarsDataset): Soccer tracking dataset with player positions
            and ball ownership information.
        formations (Union[List[str], Literal["shaw-glickman"]], optional): Formation
            templates to use. Either a list of formation names (e.g., ["4-3-3", "4-4-2"])
            or "shaw-glickman" for the alternative template set. Defaults to None
            (uses mplsoccer formations).

    Attributes:
        output (pl.DataFrame): Detected formations with columns:
            - object_id: Player ID
            - team_id: Team ID
            - position: Assigned position label (e.g., "LW", "CM", "GK")
            - formation: Formation name (e.g., "4-3-3")
            - is_attacking: Boolean indicating attacking (True) or defending (False)
            - frame_id (if every="frame"): Frame identifier
            - [segment_id] (if every != "frame"): Possession or time window identifier
        segments (pl.DataFrame, optional): When using temporal aggregation (every != "frame"),
            contains segment metadata:
            - segment_id: Unique segment identifier
            - n_frames: Number of frames in segment
            - start_timestamp / end_timestamp: Time bounds
            - start_frame_id / end_frame_id: Frame bounds

    Raises:
        ValueError: If dataset is not of type KloppyPolarsDataset.
        ImportError: If scipy is not installed (required for linear_sum_assignment).

    Example:
        >>> from unravel.soccer.dataset import KloppyPolarsDataset
        >>> from unravel.soccer.models.formations import EFPI
        >>> from kloppy import datasets
        >>>
        >>> # Load tracking data
        >>> dataset = datasets.load(provider="skillcorner", match_id="123")
        >>> soccer_data = KloppyPolarsDataset(kloppy_dataset=dataset)
        >>>
        >>> # Initialize EFPI detector
        >>> efpi = EFPI(dataset=soccer_data)
        >>>
        >>> # Detect formations per frame
        >>> efpi.fit(every="frame")
        >>> print(efpi.output)
        >>> # Shows: frame_id, object_id, position, formation, is_attacking
        >>>
        >>> # Detect formations per possession
        >>> efpi.fit(
        ...     every="possession",
        ...     change_after_possession=True,  # Re-detect when possession changes
        ...     change_threshold=0.2            # Re-detect if cost improves by 20%
        ... )
        >>> print(efpi.segments)
        >>> # Shows possession segments with start/end times
        >>>
        >>> # Detect formations per 5-minute window
        >>> efpi.fit(every="5m", substitutions="drop")
        >>>
        >>> # Use custom formation templates
        >>> efpi_custom = EFPI(
        ...     dataset=soccer_data,
        ...     formations=["4-3-3", "4-2-3-1", "3-5-2"]
        ... )
        >>> efpi_custom.fit(every="possession")

    Note:
        - Formation detection requires at least 10 outfield players per team.
          Frames with fewer players are automatically filtered out.
        - The algorithm assigns positions based on spatial distribution, not player roles.
          A player listed as a striker may be assigned "CM" if positioned centrally.
        - For per-frame detection, formations can change every frame. Use temporal
          aggregation (every="possession" or time windows) for more stable detection.
        - The cost metric measures total euclidean distance between players and template
          positions. Lower cost indicates better fit.

    See Also:
        :class:`~unravel.soccer.dataset.KloppyPolarsDataset`: Data loading and preprocessing.
        :meth:`fit`: Configure and run formation detection.
        :doc:`../tutorials/formation_detection`: Tutorial on formation analysis.
    """

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
        return pl.Struct(
            {
                Column.OBJECT_ID: pl.List(pl.String),
                Column.TEAM_ID: pl.List(pl.String),
                "position": pl.List(pl.String),
                "formation": pl.List(pl.String),
            }
        )

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
        """Detect team formations from player positions.

        Runs the EFPI formation detection algorithm on tracking data, identifying
        formations for both attacking and defending teams. Supports temporal aggregation
        to detect formations at different time scales (per-frame, per-possession, or
        custom time windows).

        The detection process:
        1. Groups data by the specified temporal unit (every)
        2. For each group, extracts attacking and defending team positions
        3. Compares positions to formation templates using optimal assignment
        4. Selects best-fit formation and assigns positional labels
        5. Handles substitutions and formation changes based on thresholds

        Args:
            start_time (pl.duration, optional): Start time for analysis window.
                Must be specified together with end_time and period_id. Defaults to None
                (processes all data).
            end_time (pl.duration, optional): End time for analysis window.
                Defaults to None.
            period_id (int, optional): Period ID to analyze (e.g., 1 for first half).
                Defaults to None.
            every (Optional[Union[str, Literal["frame", "period", "possession"]]], optional):
                Temporal aggregation level:
                - "frame": Detect formations every frame (no aggregation)
                - "possession": Detect formations per possession phase
                - "period": Detect formations per period (half)
                - Time string (e.g., "5m", "30s"): Detect formations per time window
                Defaults to "frame".
            formations (Union[List[str], Literal["shaw-glickman"]], optional):
                Formation templates to use. Either a list of formation names
                (e.g., ["4-3-3", "4-4-2", "3-5-2"]) or "shaw-glickman" for alternative
                templates. Defaults to None (uses all mplsoccer formations).
            substitutions (Literal["merge", "drop"], optional): How to handle substitutions
                within temporal windows:
                - "drop": Exclude players with shortest appearance in window
                - "merge": Average positions across substitution overlap (not yet implemented)
                Defaults to "drop".
            change_after_possession (bool, optional): Whether to re-detect formations
                when possession changes (even within the same temporal window).
                Defaults to True.
            change_threshold (float, optional): Minimum relative cost improvement (0-1)
                required to update the detected formation. For example, 0.2 means the new
                formation must have 20% lower cost to replace the current one. Helps
                stabilize detections. Defaults to None (always update).

        Returns:
            EFPI: Self, with detected formations stored in :attr:`output` and temporal
                segments in :attr:`segments`.

        Raises:
            ValueError: If start_time, end_time, and period_id are partially specified
                (must be all or none).

        Example:
            >>> # Per-frame detection (no temporal aggregation)
            >>> efpi = EFPI(dataset=soccer_data)
            >>> efpi.fit(every="frame")
            >>> print(efpi.output.head())
            >>> # Shows formation for each frame
            >>>
            >>> # Per-possession detection with stability threshold
            >>> efpi.fit(
            ...     every="possession",
            ...     change_after_possession=True,
            ...     change_threshold=0.15  # Only update if cost improves by 15%
            ... )
            >>> # Formation changes only when possession changes or cost improves significantly
            >>>
            >>> # Per-period detection (one formation per half)
            >>> efpi.fit(every="period")
            >>> # Single formation assignment for each period
            >>>
            >>> # 5-minute rolling window detection
            >>> efpi.fit(every="5m", substitutions="drop")
            >>> print(efpi.segments)
            >>> # Shows 5-minute windows with start/end times
            >>>
            >>> # Custom formations with time window
            >>> efpi.fit(
            ...     start_time=pl.duration(minutes=10),
            ...     end_time=pl.duration(minutes=20),
            ...     period_id=1,
            ...     every="possession",
            ...     formations=["4-3-3", "4-2-3-1"]
            ... )
            >>>
            >>> # Analyze formation changes during first half
            >>> efpi.fit(every="possession", period_id=1)
            >>> formation_changes = (
            ...     efpi.output
            ...     .group_by(["team_id", "possession_id"])
            ...     .agg(pl.col("formation").first())
            ... )
            >>> print(formation_changes)

        Note:
            - Per-frame detection can be noisy due to player movements. Use temporal
              aggregation (every="possession" or time windows) for more stable results.
            - The change_threshold parameter only applies when using temporal aggregation
              (every != "frame"). It prevents frequent formation updates within windows.
            - When using time windows (e.g., every="5m"), player positions are averaged
              across the window before formation detection.
            - Substitutions within windows are handled by the substitutions parameter:
              - "drop": Keeps the 11 players with longest appearances
              - "merge": Not yet implemented (will raise NotImplementedError)
            - The output DataFrame structure differs based on every:
              - "frame": Contains frame_id
              - "possession" / time windows: Contains segment_id and is_attacking
              - Use segments attribute to map segment_id back to frame ranges

        See Also:
            :class:`EFPI`: Class documentation with algorithm overview.
            :doc:`../tutorials/formation_detection`: Complete tutorial with examples.
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
                            returns_scalar=True,
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
                    (
                        [Column.GAME_ID, Column.PERIOD_ID, Column.TEAM_ID, segment_id]
                        if self._every != "period"
                        else [Column.GAME_ID, Column.PERIOD_ID, Column.TEAM_ID]
                    ),
                    maintain_order=True,
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
                    (
                        group_by_columns + [segment_id]
                        if self._every != "period"
                        else group_by_columns
                    ),
                    maintain_order=True,
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
                            returns_scalar=True,
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
