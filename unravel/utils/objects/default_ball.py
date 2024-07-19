import numpy as np
from dataclasses import dataclass


@dataclass
class DefaultBall(object):
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
        # Calculate vx, vy, and vz considering the possibility of NaN values in next_position
        vx = (
            (self.next_position3D[0] - self.position3D[0]) / 1.0
            if not (np.any(np.isnan(self.next_position)))
            else 0
        )
        vy = (
            (self.next_position3D[1] - self.position3D[1]) / 1.0
            if not (np.any(np.isnan(self.next_position)))
            else 0
        )
        vz = (
            (self.next_position3D[2] - self.position3D[2]) / 1.0
            if not (np.any(np.isnan(self.next_position)))
            else 0
        )

        # Update the velocity to include the z component
        self.velocity = np.asarray([vx, vy], dtype=float)
        self.velocity3D = np.asarray([vx, vy, vz], dtype=float)

        # Check if the velocity array has any NaN values, and if so, set velocity to zero in all dimensions
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.asarray([0.0, 0.0], dtype=float)
            self.velocity3D = np.asarray([0.0, 0.0, 0.0], dtype=float)

        # Update the speed calculation to include the z component
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
