from dataclasses import dataclass, field
from kloppy.domain import Dimension, Unit

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
        # Define coordinate ranges for the court
        self.x_dim = Dimension(min=0.0, max=self.court_length)
        self.y_dim = Dimension(min=0.0, max=self.court_width)
        # Basket is 4 feet in from the baseline and centered width-wise
        self.basket_x = self.x_dim.max - 4.0
        self.basket_y = (self.y_dim.max + self.y_dim.min) / 2.0

    def as_dict(self) -> dict:
        """Return the court dimensions and related settings as a dictionary."""
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
