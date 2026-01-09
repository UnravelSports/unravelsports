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
    """Compute pressing intensity metrics for soccer tracking data.

    Pressing Intensity quantifies the defensive pressure applied to ball carriers
    by measuring spatial coverage, defender proximity, and velocity components. The
    metric computes time-to-intercept and probability-to-intercept matrices between
    players, capturing how effectively defenders can close down passing options.

    The model outputs two matrices per frame:
    - **Time-to-Intercept (TTI)**: Time in seconds for each defender to reach each
      attacker, accounting for positions, velocities, and reaction time.
    - **Probability-to-Intercept (PTI)**: Probability (0-1) that a defender can
      successfully press each attacker, derived from TTI using a sigmoid function.

    These matrices enable analysis of:
    - Defensive compactness and coverage
    - Pressing triggers and coordination
    - Passing lane availability
    - Individual pressing effectiveness

    The implementation is based on tracking data research and extends concepts from
    pitch control and space occupation models.

    Args:
        dataset (KloppyPolarsDataset): Dataset containing soccer tracking data with
            positions, velocities, and ball ownership information.
        chunk_size (int, optional): Number of frames to process in each batch for
            memory efficiency. Defaults to 20000.

    Attributes:
        output (pl.DataFrame): Computed pressing intensity matrices with columns:
            - frame_id, period_id, timestamp: Frame identifiers
            - time_to_intercept: List[List[float]] - TTI matrix (rows × columns)
            - probability_to_intercept: List[List[float]] - PTI matrix (rows × columns)
            - columns: List[str] - Object IDs for column players (typically attackers)
            - rows: List[str] - Object IDs for row players (typically defenders)

    Raises:
        ValueError: If dataset is not of type KloppyPolarsDataset.

    Example:
        >>> from unravel.soccer.dataset import KloppyPolarsDataset
        >>> from unravel.soccer.models import PressingIntensity
        >>> from kloppy import datasets
        >>>
        >>> # Load tracking data
        >>> dataset = datasets.load(
        ...     provider="skillcorner",
        ...     match_id="123",
        ...     competition="EPL"
        ... )
        >>> soccer_data = KloppyPolarsDataset(kloppy_dataset=dataset)
        >>>
        >>> # Initialize pressing intensity model
        >>> pi = PressingIntensity(dataset=soccer_data)
        >>>
        >>> # Compute pressing intensity for all frames
        >>> pi.fit(
        ...     method="teams",           # 11x11 matrix (attackers × defenders)
        ...     ball_method="max",        # Merge ball and ball carrier
        ...     reaction_time=0.7,        # 0.7 second defender reaction time
        ...     time_threshold=1.5,       # 1.5 second pressing window
        ...     sigma=0.45                # Sigmoid steepness parameter
        ... )
        >>>
        >>> # Access results
        >>> print(pi.output)
        >>> # Shows time_to_intercept and probability_to_intercept matrices per frame
        >>>
        >>> # Compute pressing intensity for specific period
        >>> pi.fit(
        ...     start_time=pl.duration(minutes=0),
        ...     end_time=pl.duration(minutes=5),
        ...     period_id=1,
        ...     method="teams"
        ... )

    Note:
        - The model requires velocity data. Ensure your dataset has computed velocities
          via :meth:`KloppyPolarsDataset.load` with appropriate smoothing parameters.
        - Time-to-intercept assumes defenders accelerate optimally toward attackers
          from their current positions, bounded by max_player_speed.
        - Probability values near 1.0 indicate high pressing pressure; values near 0.0
          indicate low pressure or distant defenders.

    See Also:
        :class:`~unravel.soccer.dataset.KloppyPolarsDataset`: Data loading and preprocessing.
        :meth:`fit`: Configure and compute pressing intensity metrics.
        :doc:`../tutorials/pressing_intensity`: Tutorial on pressing intensity analysis.
    """
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

    @property
    def __get_return_dtype(self):
        return pl.Struct(
            {
                "time_to_intercept": pl.List(pl.List(pl.Float64)),
                "probability_to_intercept": pl.List(pl.List(pl.Float64)),
                "columns": pl.List(pl.String),
                "rows": pl.List(pl.String),
            }
        )

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
        line_method: Union[None, Literal["touchline", "byline", "all"]] = None,
    ):
        """Compute pressing intensity metrics for tracking data.

        Calculates time-to-intercept (TTI) and probability-to-intercept (PTI) matrices
        quantifying defensive pressure. For each frame, computes how quickly defenders
        can reach attackers and the likelihood of successful pressing actions.

        The computation considers:
        - Player positions and velocities
        - Reaction time delays
        - Maximum acceleration capabilities
        - Ball position and ball carrier proximity

        Args:
            start_time (pl.duration, optional): Start time for analysis window.
                Must be specified together with end_time and period_id. Defaults to None
                (processes all frames).
            end_time (pl.duration, optional): End time for analysis window.
                Defaults to None.
            period_id (int, optional): Period ID to analyze (e.g., 1 for first half).
                Defaults to None.
            speed_threshold (float, optional): Minimum player speed (m/s) to include in
                pressing calculations. Players below this threshold are masked out
                (PTI set to 0.0). Useful for analyzing active pressing vs passive coverage.
                Defaults to None (no filtering).
            reaction_time (float, optional): Defender reaction time in seconds before
                accelerating toward target. Models decision-making and perception delay.
                Defaults to 0.7 seconds.
            time_threshold (float, optional): Time window (seconds) for pressing opportunities.
                TTI values beyond this are considered low-pressure situations. Affects
                sigmoid conversion to probabilities. Defaults to 1.5 seconds.
            sigma (float, optional): Sigmoid steepness parameter for TTI → PTI conversion.
                Higher values create sharper transitions between high/low pressure.
                Defaults to 0.45.
            method (Literal["teams", "full"], optional): Matrix structure:
                - "teams": 11×11 matrix (ball-owning team × non-owning team)
                - "full": 22×22 matrix (all players × all players)
                Defaults to "teams".
            ball_method (Literal["include", "exclude", "max"], optional): Ball handling:
                - "include": Add ball as separate node (creates 11×12 or 22×23 matrix)
                - "exclude": Ignore ball entirely
                - "max": Merge ball with ball carrier using max(ball_tti, carrier_tti),
                  preserving matrix dimensions
                Defaults to "max" (recommended).
            orient (Literal["ball_owning", "pressing", "home_away", "away_home"], optional):
                Matrix orientation perspective:
                - "ball_owning": Rows = ball-owning team, Cols = non-owning team
                - "pressing": Rows = non-owning team, Cols = ball-owning team (transpose)
                - "home_away": Rows = home team, Cols = away team
                - "away_home": Rows = away team, Cols = home team
                Defaults to "ball_owning".
            line_method (Union[None, Literal["touchline", "byline", "all"]], optional):
                Reserved for future development (include pitch boundaries in calculations).
                Currently has no effect. Defaults to None.

        Returns:
            PressingIntensity: Self, with computed results stored in :attr:`output`.

        Raises:
            TypeError: If period_id is not an integer.
            ValueError: If method, ball_method, orient, or line_method have invalid values.
            TypeError: If reaction_time, speed_threshold, time_threshold, or sigma have
                invalid types.
            ValueError: If start_time, end_time, and period_id are partially specified
                (must be all or none).

        Example:
            >>> # Basic usage: compute pressing intensity for all frames
            >>> pi = PressingIntensity(dataset=soccer_data)
            >>> pi.fit(method="teams", ball_method="max")
            >>> print(pi.output.columns)
            ['frame_id', 'period_id', 'timestamp', 'time_to_intercept',
             'probability_to_intercept', 'columns', 'rows']
            >>>
            >>> # Analyze specific time window
            >>> pi.fit(
            ...     start_time=pl.duration(minutes=10),
            ...     end_time=pl.duration(minutes=15),
            ...     period_id=1,
            ...     method="teams"
            ... )
            >>>
            >>> # Filter for active pressing (players moving > 2 m/s)
            >>> pi.fit(
            ...     method="teams",
            ...     speed_threshold=2.0,
            ...     reaction_time=0.5,
            ...     time_threshold=1.0
            ... )
            >>>
            >>> # Full 22x22 matrix with ball as separate node
            >>> pi.fit(method="full", ball_method="include")
            >>>
            >>> # Extract pressing intensity for frame 1000
            >>> frame_data = pi.output.filter(pl.col("frame_id") == 1000)
            >>> tti_matrix = np.array(frame_data["time_to_intercept"][0])
            >>> pti_matrix = np.array(frame_data["probability_to_intercept"][0])
            >>> print(f"Max pressing probability: {pti_matrix.max():.2f}")

        Note:
            - Time windows (start_time, end_time, period_id) must be specified together
              or all set to None. Partial specification raises ValueError.
            - The output DataFrame contains nested lists for TTI and PTI matrices.
              Use `.to_numpy()` or indexing to extract arrays for analysis.
            - Matrix dimensions depend on method and ball_method:
              - "teams" + "max": 11×11
              - "teams" + "include": 11×12
              - "full" + "max": 22×22
              - "full" + "include": 22×23
            - Player IDs in "columns" and "rows" correspond to matrix dimensions and
              indicate which player occupies each position.

        See Also:
            :class:`PressingIntensity`: Class documentation with conceptual overview.
            :doc:`../tutorials/pressing_intensity`: Complete tutorial with visualizations.
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
        if line_method is not None and line_method not in [
            "touchline",
            "byline",
            "all",
        ]:
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
                    return_dtype=self.__get_return_dtype,
                    returns_scalar=True,
                ).alias("results")
            )
            .unnest("results")
        )

        return self
