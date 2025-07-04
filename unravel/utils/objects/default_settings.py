import numpy as np
from dataclasses import dataclass, field, fields, asdict
from typing import Union, Any, Dict

from kloppy.domain import Dimension, Unit, MetricPitchDimensions, Provider, Orientation

from ..features import (
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    PredictionLabelType,
    Pad,
)

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
        self.end_zone = self.x_dim.max - 10


@dataclass
class DefaultSettings:
    home_team_id: Union[str, int]
    away_team_id: Union[str, int]
    provider: Union[Provider, str]
    pitch_dimensions: Union[MetricPitchDimensions, AmericanFootballPitchDimensions]
    orientation: Orientation
    max_player_speed: float = 12.0
    max_ball_speed: float = 28.0
    max_player_acceleration: float = 6.0
    max_ball_acceleration: float = 13.5
    ball_carrier_threshold: float = 25.0
    frame_rate: int = 25

    def to_dict(self) -> Dict[str, Any]:
        """Convert the dataclass instance to a dictionary.

        If an attribute has a to_dict method, that method will be called.
        """
        result = {}

        for field in fields(self):
            value = getattr(self, field.name)

            # Check if the attribute has a to_dict method
            if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value

        return result
