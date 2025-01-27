from ...utils import DefaultGraphSettings

from dataclasses import dataclass, field
from kloppy.domain import Dimension, Unit
from typing import Optional

PITCH_LENGTH = 120.0
PITCH_WIDTH = 53.3


@dataclass
class AmericanFootballPitchDimensions:
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH
    standardized: bool = False
    unit: Unit = Unit.YARDS

    x_dim: Dimension = field(default_factory=lambda: Dimension(min=0, max=PITCH_LENGTH))
    y_dim: Dimension = field(default_factory=lambda: Dimension(min=0, max=PITCH_WIDTH))
    end_zone: float = field(init=False)

    def __post_init__(self):
        self.end_zone = self.x_dim.max - 10  # Calculated value


@dataclass
class AmericanFootballGraphSettings(DefaultGraphSettings):
    pitch_dimensions: AmericanFootballPitchDimensions = None
    attacking_non_qb_node_value: float = 0.1
    max_height: float = 225.0  # in cm
    min_height: float = 150.0
    max_weight: float = 200.0  # in kg
    min_weight: float = 60.0

    def __post_init__(self):
        if not isinstance(self.pitch_dimensions, AmericanFootballPitchDimensions):
            raise Exception(
                "Incorrect pitch_dimension type... Should be of type AmericanFootballPitchDimensions"
            )
        self._sport_specific_checks()

    def _sport_specific_checks(self):
        if self.attacking_non_qb_node_value > 1:
            self.attacking_non_qb_node_value = 1
        elif self.attacking_non_qb_node_value < 0:
            self.attacking_non_qb_node_value = 0
