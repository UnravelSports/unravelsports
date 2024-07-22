import numpy as np
from dataclasses import dataclass, field


@dataclass
class DefaultPlayer(object):
    fps: int
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
        dt = 1.0 / self.fps
        if not (
            np.any(np.isnan(self.next_position)) or np.any(np.isnan(self.position))
        ):
            vx = (self.next_position[0] - self.position[0]) / dt
            vy = (self.next_position[1] - self.position[1]) / dt
        else:
            vx = 0
            vy = 0

        self.velocity = np.asarray([vx, vy], dtype=float)

        # Re-check if any component of velocity is NaN and set to zero if it is
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.asarray([0.0, 0.0], dtype=float)

        self.speed = np.sqrt(vx**2 + vy**2)
