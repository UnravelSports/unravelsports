import logging
import sys

from dataclasses import dataclass, field, asdict

import polars as pl

from typing import List, Union, Dict, Literal

from kloppy.domain import TrackingDataset

from spektral.data import Graph

from ..exceptions import (
    KeyMismatchException,
)
from ..features import (
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    PredictionLabelType,
)

from .default_graph_settings import DefaultGraphSettings
from .custom_spektral_dataset import CustomSpektralDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


@dataclass(repr=True)
class DefaultGraphConverter:
    """
    Converts our dataset TrackingDataset into an internal structure

    Attributes:
        dataset (TrackingDataset): Kloppy TrackingDataset.
        labels (dict): Dict with a key per frame_id, like so {frame_id: True/False/1/0}
        graph_id (str, int): Set a single id for the whole Kloppy dataset.
        graph_ids (dict): Frame level control over graph ids.

        The graph_ids will be used to assign each graph an identifier. This identifier allows us to split the CustomSpektralDataset such that
            all graphs with the same id are either all in the test, train or validation set to avoid leakage. It is recommended to either set graph_id (int, str) as
            a match_id, or pass a dictionary into 'graph_ids' with exactly the same keys as 'labels' for more granualar control over the graph ids.
        The latter can be useful when splitting graphs by possession or sequence id. In this case the dict would be {frame_id: sequence_id/possession_id}.
        Note that sequence_id/possession_id should probably be unique for the whole dataset. Perhaps like so {frame_id: 'match_id-sequence_id'}. Defaults to None.


         boundary_correction (float): A correction factor for boundary calculations, used to correct out of bounds as a percentages (Used as 1+boundary_correction, ie 0.05). Defaults to None.
        self_loop_ball (bool): Flag to indicate if the ball node should have a self-loop. Defaults to True.
        adjacency_matrix_connect_type (AdjacencyMatrixConnectType): The type of connection used in the adjacency matrix, typically related to the ball. Defaults to AdjacenyMatrixConnectType.BALL.
        adjacency_matrix_type (AdjacencyMatrixType): The type of adjacency matrix, indicating how connections are structured, such as split by team. Defaults to AdjacencyMatrixType.SPLIT_BY_TEAM.
        label_type (PredictionLabelType): The type of prediction label used. Defaults to PredictionLabelType.BINARY.
        defending_team_node_value (float): Value between 0 and 1 to assign to the defending team nodes
        random_seed (int, bool): When a random_seed is given it will randomly shuffle an individual Graph without changing the underlying structure.
            When set to True it will shuffle every frame differently, False won't shuffle. Defaults to False.
            Adviced to set True when creating actual dataset.
        pad (bool): True pads to a total amount of 22 players and ball (so 23x23 adjacency matrix).
            It dynamically changes the edge feature padding size based on the combination of AdjacenyMatrixConnectType and AdjacencyMatrixType, and self_loop_ball
            Ie. AdjacenyMatrixConnectType.BALL and AdjacencyMatrixType.SPLIT_BY_TEAM has a maximum of 287 edges (11*11)*2 + (11+11)*2 + 1
        verbose (bool): The converter logs warnings / error messages when specific frames have no coordinates, or other missing information. False mutes all these warnings.
    """

    prediction: bool = False

    self_loop_ball: bool = False
    adjacency_matrix_connect_type: Union[
        Literal["ball"], Literal["ball_carrier"], Literal["no_connection"]
    ] = "ball"
    adjacency_matrix_type: Union[
        Literal["delaunay"],
        Literal["split_by_team"],
        Literal["dense"],
        Literal["dense_ap"],
        Literal["dense_dp"],
    ] = "split_by_team"
    label_type: Literal["binary"] = "binary"

    defending_team_node_value: float = 0.1
    random_seed: Union[bool, int] = False
    pad: bool = False
    verbose: bool = False

    label_col: str = None
    graph_id_col: str = None

    graph_frames: dict = field(init=False, repr=False, default=None)
    settings: DefaultGraphSettings = field(
        init=False, repr=False, default_factory=DefaultGraphSettings
    )
    
    feature_specs: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if hasattr(
            AdjacenyMatrixConnectType, self.adjacency_matrix_connect_type.upper()
        ):
            self.adjacency_matrix_connect_type = getattr(
                AdjacenyMatrixConnectType, self.adjacency_matrix_connect_type.upper()
            )
        else:
            raise ValueError(
                f"Invalid adjacency_matrix_connect_type: {self.adjacency_matrix_connect_type}. Should by of type 'ball', 'ball_carrier' or 'no_connection'"
            )

        if hasattr(AdjacencyMatrixType, self.adjacency_matrix_type.upper()):
            self.adjacency_matrix_type = getattr(
                AdjacencyMatrixType, self.adjacency_matrix_type.upper()
            )
        else:
            raise ValueError(
                f"Invalid adjacency_matrix_type: {self.adjacency_matrix_type}. Should be of type 'delaunay', 'split_by_team', 'dense', 'dense_ap' or 'dense_dp'"
            )

        if hasattr(PredictionLabelType, self.label_type.upper()):
            self.label_type = getattr(PredictionLabelType, self.label_type.upper())
        else:
            raise ValueError(
                f"Invalid label_type: {self.label_type}. Should be of type 'binary'"
            )

        if not isinstance(self.prediction, bool):
            raise Exception("'prediction' should be of type boolean (bool)")

        if not isinstance(self.self_loop_ball, bool):
            raise Exception("'self_loop_ball' should be of type boolean (bool)")

        if not isinstance(self.defending_team_node_value, (float, int)):
            raise Exception(
                "'defending_team_node_value' should be of type float or int"
            )

        if not isinstance(self.random_seed, (bool, int)):
            raise Exception("'random_seed' should be of type boolean (bool) or int")

        if not isinstance(self.pad, bool):
            raise Exception("'pad' should be of type boolean (bool)")

        if not isinstance(self.verbose, bool):
            raise Exception("'verbose' should be of type boolean (bool)")
        
        if not isinstance(self.feature_specs, dict):
            raise ValueError("feature_specs must be a dictionary")

    def _shuffle(self):
        raise NotImplementedError()

    def _sport_specific_checks(self):
        raise NotImplementedError(
            "No sport specific checks implementend... Make sure to check for existens of labels of some sort, and graph ids of some sort..."
        )

    def _apply_graph_settings(self):
        raise NotImplementedError()

    def _convert(self):
        raise NotImplementedError()

    def to_graph_frames(self) -> dict:
        raise NotImplementedError()

    def to_pickle(self) -> None:
        raise NotImplementedError()

    def to_spektral_graphs(self) -> List[Graph]:
        if not self.graph_frames:
            self.to_graph_frames()

        return [g.to_spektral_graph() for g in self.graph_frames]

    def to_custom_dataset(self) -> CustomSpektralDataset:
        """
        Spektral requires a spektral Dataset to load the data
        for docs see https://graphneural.network/creating-dataset/
        """
        return CustomSpektralDataset(graphs=self.to_spektral_graphs())
