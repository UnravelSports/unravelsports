import numpy as np
from dataclasses import dataclass, field
from typing import Union
from enum import Enum
from kloppy.domain import MetricPitchDimensions

from ..features import (
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    PredictionLabelType,
    Pad,
)


@dataclass
class DefaultGraphSettings:
    """
    Configuration settings for a Graph Neural Network (GNN) applied in sports analytics.

    Attributes:
        infer_ball_ownership (bool):
            Infers 'attacking_team' if no 'ball_owning_team' (Kloppy) or 'attacking_team' (List[Dict]) is provided, by finding player closest to ball using ball xyz.
        max_player_speed (float): The maximum speed of a player in meters per second. Defaults to 12.0.
        max_ball_speed (float): The maximum speed of the ball in meters per second. Defaults to 28.0.
        boundary_correction (float): A correction factor for boundary calculations, used to correct out of bounds as a percentages (Used as 1+boundary_correction, ie 0.05). Defaults to None.
        self_loop_ball (bool): Flag to indicate if the ball node should have a self-loop. Defaults to True.
        adjacency_matrix_connect_type (AdjacencyMatrixConnectType): The type of connection used in the adjacency matrix, typically related to the ball. Defaults to AdjacenyMatrixConnectType.BALL.
        adjacency_matrix_type (AdjacencyMatrixType): The type of adjacency matrix, indicating how connections are structured, such as split by team. Defaults to AdjacencyMatrixType.SPLIT_BY_TEAM.
        label_type (PredictionLabelType): The type of prediction label used. Defaults to PredictionLabelType.BINARY.
        defending_team_node_value (float): Value for the node feature when player is on defending team. Should be between 0 and 1 including. Defaults to 0.1.
        non_potential_receiver_node_value (float): Value for the node feature when player is NOT a potential receiver of a pass (when on opposing team or in possession of the ball). Should be between 0 and 1 including. Defaults to 0.1.
        random_seed (int, bool): When a random_seed is given it will randomly shuffle an individual Graph without changing the underlying structure.
            When set to True it will shuffle every frame differently, False won't shuffle. Defaults to False.
            Adviced to set True when creating actual dataset.
        pad (bool): True pads to a total amount of 22 players and ball (so 23x23 adjacency matrix).
            It dynamically changes the edge feature padding size based on the combination of AdjacenyMatrixConnectType and AdjacencyMatrixType, and self_loop_ball
            Ie. AdjacenyMatrixConnectType.BALL and AdjacencyMatrixType.SPLIT_BY_TEAM has a maximum of 287 edges (11*11)*2 + (11+11)*2 + 1
        verbose (bool): The converter logs warnings / error messages when specific frames have no coordinates, or other missing information. False mutes all these warnings.
    """

    infer_ball_ownership: bool = True
    max_player_speed: float = 12.0
    max_ball_speed: float = 28.0
    max_player_acceleration: float = None
    max_ball_acceleration: float = None
    self_loop_ball: bool = True
    adjacency_matrix_connect_type: AdjacenyMatrixConnectType = (
        AdjacenyMatrixConnectType.BALL
    )
    adjacency_matrix_type: AdjacencyMatrixType = AdjacencyMatrixType.SPLIT_BY_TEAM
    label_type: PredictionLabelType = PredictionLabelType.BINARY
    defending_team_node_value: float = 0.1
    random_seed: Union[int, bool] = False
    pad: bool = True
    verbose: bool = False

    _pitch_dimensions: int = field(init=False, repr=False, default=None)
    _pad_settings: Pad = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.defending_team_node_value > 1:
            self.defending_team_node_value = 1
        elif self.defending_team_node_value < 0:
            self.defending_team_node_value = 0

        if self.pad:
            if self.adjacency_matrix_type == AdjacencyMatrixType.DELAUNAY:
                raise NotImplementedError(
                    "Padding and Delaunay will cause corrupted Graphs, because of incorrect matrix size after make_sparse(A), where A now is A_delaunay. When using tracking data that is generally complete (10, 11 players per side) set pad=False."
                )
            self._pad_settings = self.__pad_settings()

    @property
    def pad_settings(self) -> Pad:
        return self._pad_settings

    def __pad_settings(self):
        """
        Compute maximum theoretical amount of padding for all different types of settings
        """
        n_players = 11
        n_ball = 1

        if self.adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL:
            max_ball_edges = (n_players * 2) * 2 + n_ball
        elif (
            self.adjacency_matrix_connect_type == AdjacenyMatrixConnectType.BALL_CARRIER
        ):
            max_ball_edges = 2 + n_ball
        elif (
            self.adjacency_matrix_connect_type
            == AdjacenyMatrixConnectType.NO_CONNECTION
        ):
            max_ball_edges = 0
        else:
            raise NotImplementedError()

        if self.adjacency_matrix_type == AdjacencyMatrixType.DELAUNAY:
            max_player_edges = 3 * (n_players * 2) - 6
        elif self.adjacency_matrix_type == AdjacencyMatrixType.SPLIT_BY_TEAM:
            max_player_edges = (n_players * n_players) * 2
        elif self.adjacency_matrix_type == AdjacencyMatrixType.DENSE:
            max_player_edges = (n_players + n_players) ** 2
        elif self.adjacency_matrix_type in (
            AdjacencyMatrixType.DENSE_AP,
            AdjacencyMatrixType.DENSE_DP,
        ):
            max_player_edges = n_players * n_players
        else:
            raise NotImplementedError()

        return Pad(
            max_edges=max_ball_edges + max_player_edges,
            max_nodes=(n_players * 2) + n_ball,
            n_players=n_players,
        )

    def _sport_specific_checks(self):
        raise NotImplementedError()

    def to_dict(self):
        """Custom serialization method that skips Enum fields (like 'unit') and serializes others."""

        def make_serializable(obj):
            if isinstance(obj, Enum):
                return Enum.value
            elif isinstance(obj, (int, float, str, bool, type(None), list, dict)):
                return obj
            elif isinstance(obj, MetricPitchDimensions):
                return {
                    key: make_serializable(value)
                    for key, value in obj.__dict__.items()
                    if not isinstance(value, Enum)
                }
            elif hasattr(obj, "__dict__"):
                return {
                    key: make_serializable(value) for key, value in obj.__dict__.items()
                }
            return None

        return {key: make_serializable(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data):
        """Custom deserialization method"""
        if "pitch_dimensions" in data:
            data["pitch_dimensions"] = MetricPitchDimensions(**data["pitch_dimensions"])
        return cls(**data)
