from dataclasses import dataclass

from ...utils import DefaultGraphSettings

from kloppy.domain import Dimension, Unit
from typing import Optional

PITCH_LENGTH = 120.0
PITCH_WIDTH = 53.3


@dataclass
class AmericanFootballPitchDimensions:
    x_dim = Dimension(min=0, max=PITCH_LENGTH)
    y_dim = Dimension(min=0, max=PITCH_WIDTH)
    standardized = False
    unit = Unit.YARDS

    pitch_length = PITCH_LENGTH
    pitch_width = PITCH_WIDTH
    end_zone = PITCH_LENGTH - 10


@dataclass
class AmericanFootballGraphSettings(DefaultGraphSettings):
    pitch_dimensions: AmericanFootballPitchDimensions = None
    ball_id: str = "football"
    qb_id: str = "QB"

    def __post_init__(self):
        if not isinstance(self.pitch_dimensions, AmericanFootballPitchDimensions):
            raise Exception(
                "Incorrect pitch_dimension type... Should be of type AmericanFootballPitchDimensions"
            )
