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

    u_mag = np.linalg.norm(u, axis=-1)  # velocitie of Pressing Players velocity
    v_mag = np.linalg.norm(v, axis=-1)  # velocitie of relative motion vector
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


def ray_line_intersections(positions, velocities, line_start, line_end):
    """
    Find intersections between multiple rays and a single line segment.
    
    Parameters:
    - positions: np.array of shape (n, 2) - Starting points of the rays
    - velocities: np.array of shape (n, 2) - velocitie vectors of the rays
    - line_start: np.array([x1, y1]) - Start point of the line segment
    - line_end: np.array([x2, y2]) - End point of the line segment
    
    Returns:
    - intersections: np.array of shape (n, 2) - Intersection points
    - mask: np.array of shape (n,) - Boolean mask indicating valid intersections
    """
    # Ensure we're working with numpy arrays
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    line_start = np.asarray(line_start, dtype=float)
    line_end = np.asarray(line_end, dtype=float)
    
    # Handle single position/velocitie case
    if positions.ndim == 1:
        positions = positions.reshape(1, 2)
    if velocities.ndim == 1:
        velocities = velocities.reshape(1, 2)
        
    n = positions.shape[0]
    
    # Normalize velocitie vectors
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = velocities / norms
    
    # Vector from line start to end
    line_vec = line_end - line_start
    
    # Initialize results arrays
    intersections = np.zeros((n, 2))
    mask = np.zeros(n, dtype=bool)
    
    # Check if line is vertical (x1 == x2)
    if abs(line_vec[0]) < 1e-10:
        # Check which rays have horizontal component
        horizontal_component = np.abs(velocities[:, 0]) >= 1e-10
        
        # Calculate intersection parameters
        t = np.zeros(n)
        t[horizontal_component] = (line_start[0] - positions[horizontal_component, 0]) / velocities[horizontal_component, 0]
        
        # Find valid intersections (t >= 0)
        valid = horizontal_component & (t >= 0)
        
        # Calculate intersection points
        temp_intersections = positions[valid] + t[valid, np.newaxis] * velocities[valid]
        
        # Check if intersection is within line segment bounds
        in_bounds = (min(line_start[1], line_end[1]) <= temp_intersections[:, 1]) & (temp_intersections[:, 1] <= max(line_start[1], line_end[1]))
        
        # Update valid mask
        valid_indices = np.where(valid)[0]
        final_valid = valid_indices[in_bounds]
        mask[final_valid] = True
        
        # Fill in valid intersection points
        intersections[final_valid] = positions[final_valid] + t[final_valid, np.newaxis] * velocities[final_valid]
        
        return intersections, mask
    # Check if line is horizontal (y1 == y2)
    elif abs(line_vec[1]) < 1e-10:
        # Check which rays have vertical component
        vertical_component = np.abs(velocities[:, 1]) >= 1e-10
        
        # Calculate intersection parameters
        t = np.zeros(n)
        t[vertical_component] = (line_start[1] - positions[vertical_component, 1]) / velocities[vertical_component, 1]
        
        # Find valid intersections (t >= 0)
        valid = vertical_component & (t >= 0)
        
        # Calculate intersection points
        temp_intersections = positions[valid] + t[valid, np.newaxis] * velocities[valid]
        
        # Check if intersection is within line segment bounds
        in_bounds = (min(line_start[0], line_end[0]) <= temp_intersections[:, 0]) & (temp_intersections[:, 0] <= max(line_start[0], line_end[0]))
        
        # Update valid mask
        valid_indices = np.where(valid)[0]
        final_valid = valid_indices[in_bounds]
        mask[final_valid] = True
        
        # Fill in valid intersection points
        intersections[final_valid] = positions[final_valid] + t[final_valid, np.newaxis] * velocities[final_valid]
        
        return intersections, mask
    else:
        raise NotImplementedError("Diagonal lines are not supported...")

def rotate_vectors(positions, velocities, pivots, valid_mask):
    """
    Rotate multiple vectors 180 degrees around corresponding pivot points.
    
    Parameters:
    - positions: np.array of shape (n, 2) - Positions of the vectors
    - velocities: np.array of shape (n, 2) - velocitie/velocitie vectors
    - pivots: np.array of shape (n, 2) - Pivot points for rotation
    - valid_mask: np.array of shape (n,) - Boolean mask indicating valid pivots
    
    Returns:
    - new_positions: np.array of shape (n, 2) - New positions after rotation
    - new_velocities: np.array of shape (n, 2) - New velocities after rotation
    """
    # Initialize output arrays with original values
    new_positions = positions.copy()
    new_velocities = velocities.copy()
    
    # Apply the rotation only for valid pivots
    if np.any(valid_mask):
        # For a 180-degree rotation around a point:
        # new_position = 2 * pivot - position
        new_positions[valid_mask] = 2 * pivots[valid_mask] - positions[valid_mask]
        
        # Reverse the velocitie velocitie
        new_velocities[valid_mask] = -velocities[valid_mask]
    
    return new_positions, new_velocities

def rotate_around_line(positions, velocities, line_start, line_end):
    """
    Rotate multiple vectors 180 degrees around the intersections of their extended velocities with a line.
    
    Parameters:
    - positions: np.array of shape (n, 2) - Positions of the vectors
    - velocities: np.array of shape (n, 2) - velocitie/velocitie vectors
    - line_start: np.array([x1, y1]) - Start point of the line
    - line_end: np.array([x2, y2]) - End point of the line
    
    Returns:
    - new_positions: np.array of shape (n, 2) - New positions after rotation
    - new_velocities: np.array of shape (n, 2) - New velocities after rotation
    - intersections: np.array of shape (n, 2) - Intersection points
    - valid_mask: np.array of shape (n,) - Boolean mask indicating valid intersections
    """
    # Find the intersections between the extended velocities and the line
    intersections, valid_mask = ray_line_intersections(positions, velocities, line_start, line_end)
    
    # Rotate around the intersection points
    new_positions, new_velocities = rotate_vectors(positions, velocities, intersections, valid_mask)
    
    return new_positions, new_velocities, intersections, valid_mask