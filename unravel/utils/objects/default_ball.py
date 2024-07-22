import numpy as np
from dataclasses import dataclass


@dataclass
class DefaultBall(object):
    fps: int
    x1: float = np.nan
    y1: float = np.nan
    z1: float = 0.0
    x2: float = np.nan
    y2: float = np.nan
    z2: float = 0.0
    position = np.array([np.nan, np.nan])
    position3D = np.array([np.nan, np.nan, np.nan])

    def __post_init__(self):
        self.position = np.array([self.x1, self.y1])
        self.position3D = np.array([self.x1, self.y1, self.z1])
        self.next_position = np.array([self.x2, self.y2])
        self.next_position3D = np.array([self.x2, self.y2, self.z2])

        self.set_velocity()

    def set_velocity(self):
        delta_time = 1.0 / self.fps
        
        if not (np.any(np.isnan(self.next_position3D)) or np.any(np.isnan(self.position3D))):
            vx = (self.next_position3D[0] - self.position3D[0]) / delta_time
            vy = (self.next_position3D[1] - self.position3D[1]) / delta_time
            vz = (self.next_position3D[2] - self.position3D[2]) / delta_time
        else:
            vx = 0
            vy = 0
            vz = 0

        self.velocity = np.asarray([vx, vy], dtype=float)
        self.velocity3D = np.asarray([vx, vy, vz], dtype=float)

        if np.any(np.isnan(self.velocity)):
            self.velocity = np.asarray([0.0, 0.0], dtype=float)
            self.velocity3D = np.asarray([0.0, 0.0, 0.0], dtype=float)

        self.speed = np.sqrt(vx**2 + vy**2 + vz**2)


    def invert_position(self):
        self.next_position = self.next_position * -1.0
        self.position = self.position * -1.0
        self.x1 = self.x1 * -1.0
        self.y1 = self.y1 * -1.0
        self.x2 = self.x2 * -1.0
        self.y2 = self.y2 * -1.0
        self.set_velocity()
        return self
