import numpy as np

import warnings

from typing import Union

from dataclasses import dataclass, field

from warnings import *

from spektral.data import Graph

from .features import (
    delaunay_adjacency_matrix,
    adjacency_matrix,
    node_features,
    edge_features,
)
from ...utils import (
    DefaultGraphSettings,
    DefaultTrackingModel,
    DefaultGraphFrame,
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    AdjcacenyMatrixTypeNotSetException,
)


@dataclass
class GraphFrame(DefaultGraphFrame):

    def to_spektral_graph(self) -> Graph:
        if self.graph_data:
            return Graph(
                x=self.graph_data["x"],
                a=self.graph_data["a"],
                e=self.graph_data["e"],
                y=self.graph_data["y"],
                id=self.graph_id,
            )
        else:
            return None

    def _adjaceny_matrix(self):
        """
        Create adjeceny matrices. If we specify the Adjaceny Matrix type to be Delaunay it's created as the 'general' A,
        else we create a seperate one as A_delaunay.
        This way we can use the Delaunay matrix in the Edge Features if it's not used as the Adj Matrix
        """
        if not self.settings.adjacency_matrix_type:
            raise AdjcacenyMatrixTypeNotSetException(
                "AdjacencyMatrixTypeNotSet Error... Please set `adjacency_matrix_type`..."
            )
        elif self.settings.adjacency_matrix_type == AdjacencyMatrixType.DELAUNAY:
            A = delaunay_adjacency_matrix(
                self.data.attacking_players,
                self.data.defending_players,
                self.settings.adjacency_matrix_connect_type,
                self.data.ball_carrier_idx,
                self.settings.self_loop_ball,
            )
            A_delaunay = None
        else:
            A = adjacency_matrix(
                self.data.attacking_players,
                self.data.defending_players,
                self.settings.adjacency_matrix_connect_type,
                self.settings.adjacency_matrix_type,
                self.data.ball_carrier_idx,
            )
            A_delaunay = delaunay_adjacency_matrix(
                self.data.attacking_players,
                self.data.defending_players,
                self.settings.adjacency_matrix_connect_type,
                self.data.ball_carrier_idx,
                self.settings.self_loop_ball,
            )
        return A, A_delaunay

    def _node_features(self):
        return node_features(
            attacking_players=self.data.attacking_players,
            defending_players=self.data.defending_players,
            ball=self.data.ball,
            max_player_speed=self.settings.max_player_speed,
            max_ball_speed=self.settings.max_ball_speed,
            ball_carrier_idx=self.data.ball_carrier_idx,
            pitch_dimensions=self.settings.pitch_dimensions,
            include_ball_node=True,
            defending_team_node_value=self.settings.defending_team_node_value,
            non_potential_receiver_node_value=self.settings.non_potential_receiver_node_value,
        )

    def _edge_features(self, A, A_delaunay):
        return edge_features(
            self.data.attacking_players,
            self.data.defending_players,
            self.data.ball,
            self.settings.max_player_speed,
            self.settings.max_ball_speed,
            self.settings.pitch_dimensions,
            A,
            A_delaunay,
        )
