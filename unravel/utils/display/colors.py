from dataclasses import dataclass, field
from typing import Union, Tuple

import re


@dataclass
class Color:
    color: Union[str, Tuple[int, int, int], Tuple[int, int, int, float]]
    hex_value: str = field(init=False)

    def __post_init__(self):
        self.hex_value = self.to_hex(self.color)

    @staticmethod
    def to_hex(
        color: Union[str, Tuple[int, int, int], Tuple[int, int, int, float]]
    ) -> str:
        if isinstance(color, str):
            try:
                import matplotlib.colors as mcolors
            except ImportError:
                raise ImportError(
                    "Seems like you don't have matplotlib installed. Please"
                    " install it using: pip install matplotlib"
                )
            # Handle named colors via Matplotlib
            if color.lower() in mcolors.CSS4_COLORS:
                return mcolors.to_hex(color)
            # Handle hex format
            if re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", color):
                return color.lower()
            raise ValueError(f"Invalid color format: {color}")

        elif isinstance(color, tuple):
            if len(color) == 3:
                r, g, b = color
                return f"#{r:02x}{g:02x}{b:02x}"
            elif len(color) == 4:
                r, g, b, a = color
                if not (0 <= a <= 1):
                    raise ValueError("Alpha value must be between 0 and 1.")
                return f"#{r:02x}{g:02x}{b:02x}{int(a * 255):02x}"
            else:
                raise ValueError("Tuple must be RGB or RGBA.")
        else:
            raise TypeError("Unsupported color format.")


@dataclass
class TeamColors:
    jersey: Color
    goalkeeper: Color = None

    def __post_init__(self):
        if not isinstance(self.jersey, Color):
            self.jersey = Color(self.jersey)
        if not isinstance(self.goalkeeper, Color):
            self.goalkeeper = Color(self.goalkeeper)


@dataclass
class GameColors:
    home_team: TeamColors
    away_team: TeamColors


YlRd = ["#F7FBFF", "#FFEDA0", "#FEB24C", "#FD8D3C", "#E31A1C", "#BD0026", "#800026"]


@dataclass
class ColorMaps:
    _YlRd = ["#FFFF00", "#FF0000"]  # Replace with actual YlRd values

    @property
    def YELLOW_RED(self):
        try:
            from matplotlib.colors import LinearSegmentedColormap

            return LinearSegmentedColormap.from_list("", self._YlRd)
        except ImportError:
            raise ImportError(
                "Seems like you don't have matplotlib installed. Please install it using: pip install matplotlib"
            )

    @property
    def YELLOW_RED_R(self):
        try:
            from matplotlib.colors import LinearSegmentedColormap

            return LinearSegmentedColormap.from_list("", list(reversed(self._YlRd)))
        except ImportError:
            raise ImportError(
                "Seems like you don't have matplotlib installed. Please install it using: pip install matplotlib"
            )
