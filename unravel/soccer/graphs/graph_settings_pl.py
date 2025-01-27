from dataclasses import dataclass

from ...utils import DefaultGraphSettings

from dataclasses import dataclass, field
from kloppy.domain import Dimension, Unit, MetricPitchDimensions
from typing import Optional

from .dataset import Constant


@dataclass
class GraphSettingsPolars(DefaultGraphSettings):
    ball_id: str = Constant.BALL
    goalkeeper_id: str = "GK"
    boundary_correction: float = None
    non_potential_receiver_node_value: float = 0.1
    ball_carrier_treshold: float = 25.0
    pitch_dimensions: MetricPitchDimensions = field(
        init=False, repr=False, default_factory=MetricPitchDimensions
    )

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
