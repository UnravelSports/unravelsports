import numpy as np
from dataclasses import dataclass, field


@dataclass
class DefaultPlayer(object):
    x1: float = np.nan
    y1: float = np.nan
    x2: float = np.nan
    y2: float = np.nan
    is_visible: bool = False
    position: np.array = field(
        default_factory=lambda: np.array([np.nan, np.nan], dtype=float)
    )
    next_position: np.array = field(
        default_factory=lambda: np.array([np.nan, np.nan], dtype=float)
    )

    velocity: np.array = field(
        default_factory=lambda: np.asarray([0.0, 0.0], dtype=float)
    )  # velocity vector
    speed: float = 0.0  # actual speed in m/s
    is_gk: bool = False

    def __post_init__(self):
        self.next_position = np.asarray([self.x2, self.y2], dtype=float)
        self.position = np.asarray([self.x1, self.y1], dtype=float)

        self.set_velocity()

    def invert_position(self):
        self.next_position = self.next_position * -1.0
        self.position = self.position * -1.0
        self.x1 = self.x1 * -1
        self.y1 = self.y1 * -1
        self.x2 = self.x2 * -1.0
        self.y2 = self.y2 * -1.0
        self.set_velocity()
        return self

    def set_velocity(self):
        vx = (
            (self.next_position[0] - self.position[0]) / 1.0
            if not (np.any(np.isnan(self.next_position)))
            else 0
        )
        vy = (
            (self.next_position[1] - self.position[1]) / 1.0
            if not (np.any(np.isnan(self.next_position)))
            else 0
        )

        self.velocity = np.asarray([vx, vy], dtype=float)
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.asarray([0.0, 0.0], dtype=float)

        self.speed = np.sqrt(vx**2 + vy**2)
