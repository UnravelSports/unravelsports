import numpy as np

from kloppy.domain import Orientation

import numpy as np
import polars as pl

from dataclasses import dataclass, field

from typing import List, Dict

from ...dataset.kloppy_polars import (
    KloppyPolarsDataset,
    Column,
    Constant,
)


@dataclass
class DetectedFormation:
    is_attack: bool
    formation_name: str = None
    cost: float = None
    labels: np.ndarray = field(default_factory=np.ndarray)
    ids: np.ndarray = field(default_factory=np.ndarray)

    def __post_init__(self):
        self.n_outfield_players = len(self.labels[self.labels != "GK"])
        self.labels_dict = dict(zip(self.ids, self.labels))

    def update(
        self,
        is_attack: bool,
        formation_name: str,
        cost: float,
        labels: np.ndarray = None,
        ids: np.ndarray = None,
    ):
        self.is_attack = is_attack
        self.formation_name = formation_name
        self.cost = cost

        for object_id, label in zip(ids, labels):
            self.labels_dict[object_id] = label


@dataclass
class FormationDetection:
    dataset: KloppyPolarsDataset
    chunk_size: int = field(init=True, repr=False, default=2_000)

    def __post_init__(self):
        if not isinstance(self.dataset, KloppyPolarsDataset):
            raise ValueError("dataset should be of type KloppyPolarsDataset...")

        if not self.dataset.settings.orientation == Orientation.BALL_OWNING_TEAM:
            raise ValueError(
                "KloppyPolarsDataset orientation should be Orientation.BALL_OWNING_TEAM..."
            )

        self.settings = self.dataset.settings
        self.dataset = self.dataset.data

    def __repr__(self):
        n_frames = (
            self.output[Column.FRAME_ID].n_unique() if hasattr(self, "output") else None
        )
        window_size = self._window_size if self._window_size is not None else 1
        return f"FormationDetection(n_frames={n_frames}, window_size={window_size})"

    @property
    def _exprs_variables(self):
        return [
            Column.X,
            Column.Y,
            Column.TEAM_ID,
            Column.BALL_OWNING_TEAM_ID,
            Column.OBJECT_ID,
            Column.POSITION_NAME,
        ]

    def __compute(self, args: List[pl.Series]) -> dict:
        raise NotImplementedError()

    def fit(
        self,
    ):
        raise NotImplementedError()


@dataclass
class Formations:
    pitch_length: float
    pitch_width: float
    formations: List[str] = None
    detected_formations: Dict[str, DetectedFormation] = field(init=False, repr=False)

    def __post_init__(self):
        self.detected_formations = dict()
        self._pitch()
        self.get_formations()

    def set_detected_formation(
        self,
        team_id: str,
        is_attack: bool,
        name: str,
        cost: float,
        labels: np.ndarray = None,
        ids: np.ndarray = None,
    ):
        if self.detected_formations.get(team_id, None) is None:
            self.detected_formations[team_id] = DetectedFormation(
                is_attack=is_attack,
                formation_name=name,
                cost=cost,
                labels=labels,
                ids=ids,
            )
        else:
            self.detected_formations[team_id].update(
                is_attack=is_attack,
                formation_name=name,
                cost=cost,
                labels=labels,
                ids=ids,
            )

    def get_detected_formations_as_dict(self, object_ids: list, team_ids: list):
        positions, formations = [], []

        for object_id, team_id in zip(object_ids, team_ids):

            if object_id == Constant.BALL:
                positions.append(Constant.BALL)
                formations.append(Constant.BALL)
                continue

            team_formation = self.detected_formations[team_id]
            positions.append(team_formation.labels_dict[object_id])
            formations.append(team_formation.formation_name)

        return {
            Column.OBJECT_ID: object_ids,
            Column.TEAM_ID: team_ids,
            "position": positions,
            "formation": formations,
        }

    def get_options(self):
        if self.formations is None:
            return [x for x in self.pitch.formations if not x.isalpha()]
        elif self.formations == "shaw-glickman":
            return [
                "5221",
                "352",
                "343flat",
                "3232",
                "4222",
                "41212",
                "343",
                "41221",
                "433",
                "4321",
                "4141",
                "442",
                "3331",
                "31312",
                "3241",
                "3142",
                "2422",
                "2332",
                "2431",
            ]
        else:
            return self.formations

    def _pitch(self):
        try:
            from mplsoccer import Pitch
        except ImportError:
            raise ImportError(
                "Seems like you don't have mplsoccer installed. Please"
                " install it using: pip install mplsoccer"
            )
        self.pitch = Pitch(
            pitch_type="secondspectrum",
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
        )

    def get_positions(self, formation: str):
        if formation not in self.pitch.formations:
            raise ValueError(f"Formation {formation} is not available.")
        return self.pitch.get_formation(formation)

    def get_formation_positions_left_to_right(self):
        return self._formations_coords_ltr

    def get_formation_positions_right_to_left(self):
        return self._formations_coords_rtl

    def get_formation_labels_left_to_right(self):
        return self._formations_labels_ltr

    def get_formation_labels_right_to_left(self):
        return self._formations_labels_rtl

    def get_formations(self):
        self._formations_coords_ltr = {k: dict() for k in [8, 9, 10]}
        self._formations_coords_rtl = {k: dict() for k in [8, 9, 10]}
        self._formations_labels_ltr = {k: dict() for k in [8, 9, 10]}
        self._formations_labels_rtl = {k: dict() for k in [8, 9, 10]}

        for formation in self.get_options():
            positions = self.get_positions(formation)

            f = [
                {
                    k: v
                    for k, v in pos.__dict__.items()
                    if not k in ["location", "statsbomb", "wyscout", "opta"]
                }
                for pos in positions
                if pos.name != "GK"
            ]
            labels = np.asarray([pos.name for pos in positions if pos.name != "GK"])
            self._formations_coords_ltr[len(f)][formation] = np.array(
                [(v["x"], v["y"]) for v in f]
            )
            self._formations_coords_rtl[len(f)][formation] = np.array(
                [(v["x_flip"], v["y_flip"]) for v in f]
            )
            self._formations_labels_ltr[len(f)][formation] = labels
            self._formations_labels_rtl[len(f)][formation] = labels
