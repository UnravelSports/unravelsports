import numpy as np
from dataclasses import dataclass, field
from typing import Union
from enum import Enum

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
    
    def to_dict(self):
        """Custom serialization method that skips Enum fields (like 'unit') and serializes others."""
        
        def make_serializable(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (int, float, str, bool, type(None), list, dict)):
                return obj
            elif isinstance(obj, MetricPitchDimensions):  
                return {
                    key: make_serializable(value)
                    for key, value in obj.__dict__.items()
                    if not isinstance(value, Enum)
                }
            elif hasattr(obj, "__dict__"):  
                return {key: make_serializable(value) for key, value in obj.__dict__.items()}
            return None

        return {key: make_serializable(value) for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data):
        """Custom deserialization method"""
        if "pitch_dimensions" in data:
            data["pitch_dimensions"] = MetricPitchDimensions(**data["pitch_dimensions"])
        return cls(**data)