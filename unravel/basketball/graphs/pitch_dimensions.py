class BasketballPitchDimensions:
    """
    Defines the physical dimensions of a basketball court.
    Default values are based on official NBA court dimensions.
    """
    def __init__(self, court_length: float = 94.0, court_width: float = 50.0, three_point_radius: float = 23.75):
        # Court dimensions in feet.
        self.court_length = court_length
        self.court_width = court_width
        # Three-point line radius (assumes uniform for simplification).
        self.three_point_radius = three_point_radius
        # Assume the basket is 4 feet in from the baseline.
        self.basket_x = court_length - 4.0
        self.basket_y = court_width / 2

    def as_dict(self) -> dict:
        """Return the court dimensions as a dictionary."""
        return {
            "court_length": self.court_length,
            "court_width": self.court_width,
            "three_point_radius": self.three_point_radius,
            "basket_x": self.basket_x,
            "basket_y": self.basket_y,
        }
