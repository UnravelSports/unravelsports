from ...utils import DefaultGraphSettings, AmericanFootballPitchDimensions

from dataclasses import dataclass, field
from kloppy.domain import Dimension, Unit
from typing import Optional


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
