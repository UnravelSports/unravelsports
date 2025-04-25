from dataclasses import dataclass, field
from typing import Union

from kloppy.domain import Dimension, Unit

from unravel.utils.objects.default_graph_settings import DefaultGraphSettings
from unravel.utils.features import (
    AdjacencyMatrixType,
    AdjacenyMatrixConnectType,
    PredictionLabelType,
)

@dataclass
class BasketballPitchDimensions:
    """
    Defines the physical dimensions of a basketball court.
    Default values are based on official NBA court dimensions.
    """
    court_length: float = 94.0
    court_width: float = 50.0
    three_point_radius: float = 23.75

    standardized: bool = False
    unit: Unit = Unit.FEET

    x_dim: Dimension = field(init=False)
    y_dim: Dimension = field(init=False)
    basket_x: float = field(init=False)
    basket_y: float = field(init=False)

    def __post_init__(self):
        self.x_dim = Dimension(min=0.0, max=self.court_length)
        self.y_dim = Dimension(min=0.0, max=self.court_width)
        self.basket_x = self.x_dim.max - 4.0
        self.basket_y = (self.y_dim.max + self.y_dim.min) / 2.0

    def as_dict(self) -> dict:
        return {
            "court_length": self.court_length,
            "court_width": self.court_width,
            "three_point_radius": self.three_point_radius,
            "standardized": self.standardized,
            "unit": self.unit,
            "x_dim": {"min": self.x_dim.min, "max": self.x_dim.max},
            "y_dim": {"min": self.y_dim.min, "max": self.y_dim.max},
            "basket_x": self.basket_x,
            "basket_y": self.basket_y,
        }

@dataclass(kw_only=True)
class BasketballGraphSettings(DefaultGraphSettings):
    """
    Configuration settings for converting NBA tracking data into graph representations.
    Inherits from DefaultGraphSettings to leverage common functionality.
    """
    # Embed pitch dimensions with default factory for convenience
    pitch_dimensions: BasketballPitchDimensions = field(default_factory=BasketballPitchDimensions)

    # Basketball-specific overrides
    ball_carrier_threshold: float = 5.0
    defending_team_node_value: float = 0.0
    attacking_team_node_value: float = 1.0

    def __post_init__(self):
        # Ensure correct type
        if not isinstance(self.pitch_dimensions, BasketballPitchDimensions):
            raise TypeError("pitch_dimensions must be a BasketballPitchDimensions instance")
        # Validate node-value bounds
        for name, val in [
            ("defending_team_node_value", self.defending_team_node_value),
            ("attacking_team_node_value", self.attacking_team_node_value),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")
        # Call base class checks (self_loop, adjacency types, labels, etc.)
        super().__post_init__()
