import numpy as np
from dataclasses import dataclass, field
from typing import Union
from kloppy.domain import MetricPitchDimensions

from ..features import (
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    PredictionLabelType,
    Pad,
)


@dataclass
class GraphSettings:
    """
    Configuration settings for a Graph Neural Network (GNN) applied in sports analytics.

    Attributes:
        infer_ball_ownership (bool):
            Infers 'attacking_team' if no 'ball_owning_team' (Kloppy) or 'attacking_team' (List[Dict]) is provided, by finding player closest to ball using ball xyz.
            Also infers ball_carrier within ball_carrier_threshold
        infer_goalkeepers (bool): set True if no GK label is provider, set False for incomplete (broadcast tracking) data that might not have a GK in every frame
        ball_carrier_threshold (float): The distance threshold to determine the ball carrier. Defaults to 25.0.
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
    infer_goalkeepers: bool = True
    ball_carrier_treshold: float = 25.0
    max_player_speed: float = 12.0
    max_ball_speed: float = 28.0
    boundary_correction: float = None
    self_loop_ball: bool = True
    adjacency_matrix_connect_type: AdjacenyMatrixConnectType = (
        AdjacenyMatrixConnectType.BALL
    )
    adjacency_matrix_type: AdjacencyMatrixType = AdjacencyMatrixType.SPLIT_BY_TEAM
    label_type: PredictionLabelType = PredictionLabelType.BINARY
    defending_team_node_value: float = 0.1
    non_potential_receiver_node_value: float = 0.1
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

        if self.non_potential_receiver_node_value > 1:
            self.non_potential_receiver_node_value = 1
        elif self.non_potential_receiver_node_value < 0:
            self.non_potential_receiver_node_value = 0

        if self.pad:
            if self.adjacency_matrix_type == AdjacencyMatrixType.DELAUNAY:
                raise NotImplementedError(
                    "Padding and Delaunay will cause corrupted Graphs, because of incorrect matrix size after make_sparse(A), where A now is A_delaunay. When using tracking data that is generally complete (10, 11 players per side) set pad=False."
                )
            self._pad_settings = self.__pad_settings()

    @property
    def pad_settings(self) -> Pad:
        return self._pad_settings

    @property
    def pitch_dimensions(self) -> int:
        return self._pitch_dimensions

    @pitch_dimensions.setter
    def pitch_dimensions(self, pitch_dimensions: MetricPitchDimensions) -> None:
        self._pitch_dimensions = pitch_dimensions

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
            AdjacencyMatrixType.DENSE_ATTACKING_PLAYERS,
            AdjacencyMatrixType.DENSE_DEFENSIVE_PLAYERS,
        ):
            max_player_edges = n_players * n_players
        else:
            raise NotImplementedError()

        return Pad(
            max_edges=max_ball_edges + max_player_edges,
            max_nodes=(n_players * 2) + n_ball,
            n_players=n_players,
        )

    def to_dict(self):
        return {
            "infer_ball_ownership": self.infer_ball_ownership,
            "infer_goalkeepers": self.infer_goalkeepers,
            "ball_carrier_treshold": self.ball_carrier_treshold,
            "max_player_speed": self.max_player_speed,
            "max_ball_speed": self.max_ball_speed,
            "boundary_correction": self.boundary_correction,
            "self_loop_ball": self.self_loop_ball,
            "adjacency_matrix_connect_type": self.adjacency_matrix_connect_type,
            "adjacency_matrix_type": self.adjacency_matrix_type,
            "label_type": self.label_type,
            "defending_team_node_value": self.defending_team_node_value,
            "non_potential_receiver_node_value": self.non_potential_receiver_node_value,
            "random_seed": self.random_seed,
            "pad": self.pad,
            "verbose": self.verbose,
            "pitch_dimensions": self._serialize_pitch_dimensions(),
            "pad_settings": self.pad_settings
        }
        
    def _serialize_pitch_dimensions(self):
        return {
            "pitch_length": self.pitch_dimensions.pitch_length,
            "pitch_width": self.pitch_dimensions.pitch_width,
            "max_x": self.pitch_dimensions.x_dim.max,
            "min_x": self.pitch_dimensions.x_dim.min,
            "max_y": self.pitch_dimensions.y_dim.max,
            "min_y": self.pitch_dimensions.y_dim.min,
        }