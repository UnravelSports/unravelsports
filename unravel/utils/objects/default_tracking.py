import numpy as np
from typing import Union, Dict, List

from kloppy.domain import (
    TrackingDataset,
    Frame,
    Point3D,
    Point,
    Orientation,
    Ground,
    AttackingDirection,
)

from . import DefaultPlayer, DefaultBall
from ..exceptions import (
    InvalidAttackingTeamType,
    MissingAttackingTeam,
    MissingCoordinates,
    NoNextFrameWarning,
)

import warnings

from dataclasses import dataclass, field


@dataclass
class DefaultTrackingModel:
    frame: TrackingDataset

    orientation: Orientation
    infer_ball_ownership: bool = False
    infer_goalkeepers: bool = False
    ball_carrier_treshold: bool = 25.0
    verbose: bool = False
    pad_n_players: bool = None

    def __post_init__(self):
        self.home_players: List[DefaultPlayer] = list()
        self.away_players: List[DefaultPlayer] = list()
        self.ball: DefaultBall = DefaultBall()
        self.attacking_team: str = None
        self.ball_carrier_idx: int = None

        self.set_objects_from_frame(
            infer_ball_ownership=self.infer_ball_ownership,
            infer_goalkeepers=self.infer_goalkeepers,
            ball_carrier_treshold=self.ball_carrier_treshold,
            orientation=self.orientation,
            verbose=self.verbose,
            pad_n_players=self.pad_n_players,
        )

    @property
    def attacking_players(self):
        if self.attacking_team is None:
            warnings.warn(
                """No key 'attacking_team' found in 'Frame'.""", MissingAttackingTeam
            )
            return self.home_players
        if not self.attacking_team in [
            Ground.HOME,
            Ground.AWAY,
            Ground.HOME.value,
            Ground.AWAY.value,
        ]:
            raise InvalidAttackingTeamType(
                f"'attacking_team' should be of type {Ground.HOME} or {Ground.AWAY}"
            )

        return (
            self.home_players
            if self.attacking_team in [Ground.HOME, Ground.HOME.value]
            else self.away_players
        )

    @property
    def defending_players(self):
        if self.attacking_team is None:
            warnings.warn(
                """No key 'attacking_team' found in 'Frame'. """, MissingAttackingTeam
            )
            return self.away_players
        if not self.attacking_team in [
            Ground.HOME,
            Ground.AWAY,
            Ground.HOME.value,
            Ground.AWAY.value,
        ]:
            raise InvalidAttackingTeamType(
                f"'attacking_team' should be of type {Ground.HOME} or {Ground.AWAY}"
            )

        return (
            self.home_players
            if self.attacking_team in [Ground.AWAY, Ground.AWAY.value]
            else self.away_players
        )

    def _distance_to_ball(self, players):
        """
        Use 3D distance to compute distance to the ball. Since we don't have player 3D position, we pad it with 0s
        """
        player_positions = np.asarray([p.position for p in players])
        ball_carrier_dist = np.linalg.norm(
            np.pad(player_positions, ((0, 0), (0, 1)), "constant")
            - self.ball.position3D,
            axis=1,
        )
        return ball_carrier_dist

    def _set_ball_carrier_idx(self, threshold):
        if not self.ball_carrier_idx:
            ball_carrier_dist = self._distance_to_ball(players=self.attacking_players)
            self.ball_carrier_idx = (
                np.nanargmin(ball_carrier_dist)
                if np.nanmin(ball_carrier_dist) < threshold
                else None
            )
            return self.ball_carrier_idx
        return self.ball_carrier_idx

    def _set_goalkeeper(self, players, func):
        idx = func([p.y1 for p in players])
        players[idx].is_gk = True
        return players

    def _set_attacking_team(self, threshold):
        home_ball_carrier_dists = self._distance_to_ball(self.home_players)
        closest_home_player_dist = np.nanmin(home_ball_carrier_dists)
        away_ball_carrier_dists = self._distance_to_ball(self.away_players)
        closest_away_player_dist = np.nanmin(away_ball_carrier_dists)

        if (
            np.nanmin(
                np.concatenate((home_ball_carrier_dists, away_ball_carrier_dists))
            )
            < threshold
        ):
            if closest_home_player_dist < closest_away_player_dist:
                self.attacking_team = Ground.HOME
                self.ball_carrier_idx = np.nanargmin(home_ball_carrier_dists)
            else:
                self.attacking_team = Ground.AWAY
                self.ball_carrier_idx = np.nanargmin(away_ball_carrier_dists)

    def set_objects_from_frame(
        self,
        infer_ball_ownership: bool = False,
        infer_goalkeepers: bool = False,
        ball_carrier_treshold: float = 25.0,
        orientation: Orientation = Orientation.NOT_SET,
        verbose: bool = True,
        pad_n_players: int = None,
    ) -> tuple[List[DefaultPlayer], List[DefaultPlayer], DefaultBall, str]:
        frame = self.frame

        if isinstance(frame, Frame):
            fix_orientation_ltr = (
                True
                if orientation == Orientation.STATIC_HOME_AWAY and infer_ball_ownership
                else False
            )

            next_frame = frame.next()

            if not next_frame:
                if verbose:
                    warnings.warn(
                        f"""No next_frame found, skipping...""", NoNextFrameWarning
                    )
                return None, None, None, None
            if not frame.ball_coordinates or not next_frame.ball_coordinates:
                if verbose:
                    warnings.warn(
                        f"""No ball_coordinates found in frame_id={frame.frame_id}, skipping...""",
                        MissingCoordinates,
                    )
                return None, None, None, None
            if not frame.players_coordinates:
                if verbose:
                    warnings.warn(
                        f"""No player_coordinates found in frame_id={frame.frame_id}, skipping...""",
                        MissingCoordinates,
                    )
                return None, None, None, None

            for pid in frame.players_data:
                coords = frame.players_data[pid].coordinates
                try:
                    if pid.positions.at_start() in ["Goalkeeper", "GK", "TW"]:
                        player.is_gk = True
                except KeyError:  # catching Kloppy key error for empty TimeContainer
                    pass

                if not pid in next_frame.players_data:
                    continue

                next_coords = next_frame.players_data[pid].coordinates

                if coords is not None:
                    player = DefaultPlayer(
                        x1=coords.x,
                        x2=next_coords.x,
                        y1=coords.y,
                        y2=next_coords.y,
                        is_visible=True,
                    )

                    if pid.team.ground == Ground.HOME:
                        self.home_players.append(player)
                    elif pid.team.ground == Ground.AWAY:
                        self.away_players.append(player)
                    else:
                        continue

            if not self.home_players or not self.away_players:
                return

            if pad_n_players:
                for _ in range(0, pad_n_players - len(self.home_players)):
                    self.home_players.append(DefaultPlayer())
                for _ in range(0, pad_n_players - len(self.away_players)):
                    self.away_players.append(DefaultPlayer())

            if isinstance(frame.ball_coordinates, Point):
                z1, z2 = 0.0, 0.0
            elif isinstance(frame.ball_coordinates, Point3D):
                z1, z2 = frame.ball_coordinates.z, next_frame.ball_coordinates.z

            self.ball = DefaultBall(
                x1=frame.ball_coordinates.x,
                y1=frame.ball_coordinates.y,
                z1=z1,
                x2=next_frame.ball_coordinates.x,
                y2=next_frame.ball_coordinates.y,
                z2=z2,
            )

            self.attacking_team = (
                None
                if frame.ball_owning_team is None
                else frame.ball_owning_team.ground
            )
            if infer_ball_ownership:
                if not self.attacking_team:
                    self._set_attacking_team(threshold=ball_carrier_treshold)
                else:
                    self._set_ball_carrier_idx(threshold=ball_carrier_treshold)

            attacking_direction = (
                AttackingDirection.LTR
                if (
                    (orientation == Orientation.BALL_OWNING_TEAM)
                    or (
                        orientation == Orientation.STATIC_HOME_AWAY
                        and self.attacking_team == Ground.HOME
                    )
                )
                else AttackingDirection.NOT_SET
            )

            if fix_orientation_ltr and self.attacking_team == Ground.AWAY:
                self.home_players = [p.invert_position() for p in self.home_players]
                self.away_players = [p.invert_position() for p in self.away_players]
                self.ball = self.ball.invert_position()
                attacking_direction = AttackingDirection.LTR

            if infer_goalkeepers and attacking_direction == AttackingDirection.LTR:
                if not any([p.is_gk for p in self.home_players]):
                    self.home_players = self._set_goalkeeper(
                        self.home_players,
                        func=(
                            np.argmin
                            if self.attacking_team == Ground.HOME
                            else np.argmax
                        ),
                    )
                if not any([p.is_gk for p in self.away_players]):
                    self.away_players = self._set_goalkeeper(
                        self.away_players,
                        func=(
                            np.argmin
                            if self.attacking_team == Ground.AWAY
                            else np.argmax
                        ),
                    )
        else:
            raise NotImplementedError(
                """'data' dtype is not supported. Make sure it's a kloppy 'Frame'"""
            )
