from dataclasses import dataclass, field
from typing import Union

from unravel.utils.objects.default_graph_settings import DefaultGraphSettings
from unravel.utils.features import (
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    PredictionLabelType,
)
from .pitch_dimensions import BasketballPitchDimensions

@dataclass(kw_only=True)
class BasketballGraphSettings(DefaultGraphSettings):
    """
    Configuration settings for converting NBA tracking data into graph representations.
    Inherits from DefaultGraphSettings to leverage common functionality.
    """
    # Sport-specific settings
    pitch_dimensions: BasketballPitchDimensions
    ball_carrier_threshold: float = 5.0
    defending_team_node_value: float = 0.0
    attacking_team_node_value: float = 1.0

    def __post_init__(self):
        # Validate pitch_dimensions type
        if not isinstance(self.pitch_dimensions, BasketballPitchDimensions):
            raise TypeError("pitch_dimensions must be a BasketballPitchDimensions instance")
        # Validate node value ranges
        if not (0.0 <= self.defending_team_node_value <= 1.0):
            raise ValueError("defending_team_node_value must be between 0 and 1")
        if not (0.0 <= self.attacking_team_node_value <= 1.0):
            raise ValueError("attacking_team_node_value must be between 0 and 1")
        # Perform default settings validation
        super().__post_init__()