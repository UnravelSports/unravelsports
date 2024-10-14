import logging
import sys
from copy import deepcopy

import warnings

from dataclasses import dataclass, field, asdict

from typing import List, Union, Dict, Literal

from kloppy.domain import (
    TrackingDataset,
    Frame,
    Orientation,
    DatasetTransformer,
    DatasetFlag,
    SecondSpectrumCoordinateSystem,
)

from spektral.data import Graph

from .exceptions import (
    MissingLabelsError,
    MissingDatasetError,
    IncorrectDatasetTypeError,
    KeyMismatchError,
)

from .graph_settings import GraphSettings
from .graph_frame import GraphFrame

from ...utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


@dataclass(repr=True)
class SoccerGraphConverter(DefaultGraphConverter):
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
        random_seed (int, bool): When a random_seed is given it will randomly shuffle an individual Graph without changing the underlying structure.
            When set to True it will shuffle every frame differently, False won't shuffle. Defaults to False.
            Adviced to set True when creating actual dataset.
        pad (bool): True pads to a total amount of 22 players and ball (so 23x23 adjacency matrix).
            It dynamically changes the edge feature padding size based on the combination of AdjacenyMatrixConnectType and AdjacencyMatrixType, and self_loop_ball
            Ie. AdjacenyMatrixConnectType.BALL and AdjacencyMatrixType.SPLIT_BY_TEAM has a maximum of 287 edges (11*11)*2 + (11+11)*2 + 1
        verbose (bool): The converter logs warnings / error messages when specific frames have no coordinates, or other missing information. False mutes all these warnings.
    """

    dataset: TrackingDataset = None
    labels: dict = None

    labels: dict = None
    graph_id: Union[str, int, dict] = None
    graph_ids: dict = None

    infer_goalkeepers: bool = True
    infer_ball_ownership: bool = True
    boundary_correction: float = None

    def __post_init__(self):
        if not self.dataset:
            raise Exception("Please provide a 'kloppy' dataset.")

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

        self._sport_specific_checks()
        self.settings = GraphSettings(
            ball_carrier_treshold=self.ball_carrier_treshold,
            max_player_speed=self.max_player_speed,
            max_ball_speed=self.max_ball_speed,
            boundary_correction=self.boundary_correction,
            self_loop_ball=self.self_loop_ball,
            adjacency_matrix_connect_type=self.adjacency_matrix_connect_type,
            adjacency_matrix_type=self.adjacency_matrix_type,
            label_type=self.label_type,
            infer_ball_ownership=self.infer_ball_ownership,
            infer_goalkeepers=self.infer_goalkeepers,
            defending_team_node_value=self.defending_team_node_value,
            non_potential_receiver_node_value=self.non_potential_receiver_node_value,
            random_seed=self.random_seed,
            pad=self.pad,
            verbose=self.verbose,
        )

        if isinstance(self.dataset, TrackingDataset):
            if not self.dataset.metadata.flags & DatasetFlag.BALL_OWNING_TEAM:
                to_orientation = Orientation.STATIC_HOME_AWAY
            else:
                to_orientation = Orientation.BALL_OWNING_TEAM

            self.dataset = DatasetTransformer.transform_dataset(
                dataset=self.dataset,
                to_orientation=to_orientation,
                to_coordinate_system=SecondSpectrumCoordinateSystem(
                    pitch_length=self.dataset.metadata.pitch_dimensions.pitch_length,
                    pitch_width=self.dataset.metadata.pitch_dimensions.pitch_width,
                ),
            )
            self.orientation = self.dataset.metadata.orientation

        self.settings.pitch_dimensions = self.dataset.metadata.pitch_dimensions

    def _sport_specific_checks(self):
        if not self.labels and not self.prediction:
            raise Exception(
                "Please specify 'labels' or set 'prediction=True' if you want to use the converted dataset to make predictions on."
            )

        if self.graph_id is not None and self.graph_ids:
            raise Exception("Please set either 'graph_id' or 'graph_ids', not both...")

        if self.graph_ids:
            if not self.prediction:
                if not set(list(self.labels.keys())) == set(
                    list(self.graph_ids.keys())
                ):
                    raise KeyMismatchException(
                        "When 'graph_id' is of type dict it needs to have the exact same keys as 'labels'..."
                    )
        if not self.graph_ids and self.prediction:
            self.graph_ids = {x.frame_id: x.frame_id for x in self.dataset}

    def _convert(self, frame: Frame):
        data = DefaultTrackingModel(
            frame,
            fps=self.dataset.metadata.frame_rate,
            infer_ball_ownership=self.settings.infer_ball_ownership,
            infer_goalkeepers=self.settings.infer_goalkeepers,
            ball_carrier_treshold=self.settings.ball_carrier_treshold,
            orientation=self.orientation,
            verbose=self.settings.verbose,
            pad_n_players=(
                None if not self.settings.pad else self.settings.pad_settings.n_players
            ),
        )

        if isinstance(frame, Frame):
            if not self.prediction:
                label = self.labels.get(frame.frame_id, None)
            else:
                label = -1

            graph_id = None
            if (
                self.graph_id is None and not self.graph_ids
            ):  # technically graph_id can be 0
                graph_id = None
            elif self.graph_ids:
                graph_id = self.graph_ids.get(frame.frame_id, None)
            elif self.graph_id:
                graph_id = self.graph_id
            else:
                raise NotImplementedError()

            if not self.prediction and label is None:
                if self.settings.verbose:
                    warnings.warn(
                        f"""No label for frame={frame.frame_id} in 'labels'...""",
                        NoLabelWarning,
                    )
            frame_id = frame.frame_id
        else:
            raise NotImplementedError(
                """Format is not supported, should be TrackingDataset (Kloppy)"""
            )

        return data, label, frame_id, graph_id

    def to_graph_frames(self) -> dict:
        if not self.graph_frames:
            from tqdm import tqdm

            if not self.dataset:
                raise MissingDatasetError(
                    "Please specificy a 'dataset' a Kloppy TrackingDataset (see README)"
                )

            if isinstance(self.dataset, TrackingDataset):
                if not self.labels and not self.prediction:
                    raise MissingLabelsError(
                        "Please specificy 'labels' of type Dict when using Kloppy"
                    )
            else:
                raise IncorrectDatasetTypeError(
                    "dataset should be of type TrackingDataset"
                )

            self.graph_frames = list()

            for frame in tqdm(self.dataset, desc="Processing frames"):
                data, label, frame_id, graph_id = self._convert(frame)
                if data.home_players and data.away_players:
                    gnn_frame = GraphFrame(
                        frame_id=frame_id,
                        data=data,
                        label=label,
                        graph_id=graph_id,
                        settings=self.settings,
                    )
                    if gnn_frame.graph_data:
                        self.graph_frames.append(gnn_frame)

        return self.graph_frames

    def to_spektral_graphs(self) -> List[Graph]:
        if not self.graph_frames:
            self.to_graph_frames()

        return [g.to_spektral_graph() for g in self.graph_frames]

    def to_pickle(self, file_path: str) -> None:
        """
        We store the 'dict' version of the Graphs to pickle each graph is now a dict with keys x, a, e, and y
        To use for training with Spektral feed the loaded pickle data to CustomDataset(data=pickled_data)
        """
        if not file_path.endswith("pickle.gz"):
            raise ValueError(
                "Only compressed pickle files of type 'some_file_name.pickle.gz' are supported..."
            )

        if not self.graph_frames:
            self.to_graph_frames()

        import pickle
        import gzip
        from pathlib import Path

        path = Path(file_path)

        directories = path.parent
        directories.mkdir(parents=True, exist_ok=True)

        with gzip.open(file_path, "wb") as file:
            data = [x.graph_data for x in self.graph_frames]
            pickle.dump(data, file)
