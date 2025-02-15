import numpy as np


def probability_to_intercept(
    time_to_intercept: np.ndarray, tti_sigma: float, tti_time_threshold: float
):
    exponent = (
        -np.pi / np.sqrt(3.0) / tti_sigma * (tti_time_threshold - time_to_intercept)
    )
    # we take the below step to avoid Overflow errors, np.exp does not like values above ~700.
    # exp(25) should already result in p ~ 0.000%
    exponent = np.clip(exponent, -700, 700)
    p = 1 / (1.0 + np.exp(exponent))
    return p


def time_to_intercept(
    p1: np.ndarray,
    p2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reaction_time: float,
    max_object_speed: float,
) -> np.ndarray:
    """
    BSD 3-Clause License

    Copyright (c) 2025 [UnravelSports]

    See: https://opensource.org/licenses/BSD-3-Clause

    This project includes code and contributions from:
        - Joris Bekkers (UnravelSports)

    Permission is hereby granted to redistribute this software under the BSD 3-Clause License, with proper attribution
    ----------

    Calculate the Time-to-Intercept (TTI) pressing intensity for a group of players.

    This function estimates the time required for Player 1 to press Player 2 based on their
    positions, velocities, reaction times, and maximum running speed. It calculates an
    interception time matrix for all possible pairings of players.

    Parameters
    ----------
    p1 : ndarray
        An array of shape (n, 2) representing the positions of Pressing Players.
        Each row corresponds to a player's position as (x, y) coordinates.

    p2 : ndarray
        An array of shape (m, 2) representing the positions of Players on the In Possession Team (potentially including the ball location)
        Each row corresponds to a player's position as (x, y) coordinates.

    v1 : ndarray
        An array of shape (n, 2) representing the velocities corresponding to v1. Each row corresponds
        to a player's velocity as (vx, vy).

    v2 : ndarray
        An array of shape (m, 2) representing the velocities corresponding to p2. Each row corresponds
        to a player's velocity as (vx, vy).

    reaction_time : float
        The reaction time of p1'ss (in seconds) before they start moving towards p2's.

    max_velocity : float
        The maximum running velocity of Player 1 (in meters per second).

    Returns
    -------
    t : ndarray
        A 2D array of shape (m, n) where t[i, j] represents the time required for Player 1[j]
        to press Player 2[i].
    """
    u = (p1 + v1) - p1  # Adjusted velocity of Pressing Players
    d2 = p2 + v2  # Destination of Players Under Pressure

    v = (
        d2[:, None, :] - p1[None, :, :]
    )  # Relative motion vector between Pressing Players and Players Under Pressure

    u_mag = np.linalg.norm(u, axis=-1)  # Magnitude of Pressing Players velocity
    v_mag = np.linalg.norm(v, axis=-1)  # Magnitude of relative motion vector
    dot_product = np.sum(u * v, axis=-1)

    epsilon = 1e-10  # We add epsilon to avoid dividing by zero (which throws a warning)
    angle = np.arccos(dot_product / (u_mag * v_mag + epsilon))

    r_reaction = (
        p1 + v1 * reaction_time
    )  # Adjusted position of Pressing Players after reaction time
    d = d2[:, None, :] - r_reaction[None, :, :]  # Distance vector after reaction time

    t = (
        u_mag * angle / np.pi  # Time contribution from angular adjustment
        + reaction_time  # Add reaction time
        + np.linalg.norm(d, axis=-1) / max_object_speed
    )  # Time contribution from running

    return t
