from dataclasses import dataclass

from ...utils import DefaultGraphSettings

from dataclasses import dataclass, field
from kloppy.domain import Dimension, Unit, MetricPitchDimensions
from typing import Optional


@dataclass
class GraphSettingsPL(DefaultGraphSettings):
    ball_id: str = "ball"
    goalkeeper_id: str = "GK"
    infer_goalkeepers: bool = True
    boundary_correction: float = None
    non_potential_receiver_node_value: float = 0.1
    ball_carrier_treshold: float = 25.0
    pitch_dimensions: MetricPitchDimensions = field(
        init=False, repr=False, default_factory=MetricPitchDimensions
    )
    _identifier_column: str = field(default="id", init=False)

    def __post_init__(self):
        self._sport_specific_checks()

    @property
    def pitch_dimensions(self) -> int:
        return self._pitch_dimensions

    @pitch_dimensions.setter
    def pitch_dimensions(self, pitch_dimensions: MetricPitchDimensions) -> None:
        self._pitch_dimensions = pitch_dimensions

    def _sport_specific_checks(self):
        if self.non_potential_receiver_node_value > 1:
            self.non_potential_receiver_node_value = 1
        elif self.non_potential_receiver_node_value < 0:
            self.non_potential_receiver_node_value = 0
