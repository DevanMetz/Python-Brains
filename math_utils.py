def vectorized_line_circle_intersection(p1s, p2s, centers, radii, xp):
    """
    Calculates intersections between N line segments and M circles in a vectorized manner.
    This is a GPU-compatible, vectorized implementation of the geometric problem.

    Args:
        p1s (xp.ndarray): Array of line start points, shape (N, 2).
        p2s (xp.ndarray): Array of line end points, shape (N, 2).
        centers (xp.ndarray): Array of circle centers, shape (M, 2).
        radii (xp.ndarray): Array of circle radii, shape (M,).
        xp (module): The numpy or cupy module to use for calculations.

    Returns:
        tuple: A tuple containing:
            - min_distances (xp.ndarray): An array of shape (N,) with the minimum intersection
                                          distance for each line. Infinity if no intersection.
            - min_indices (xp.ndarray): An array of shape (N,) with the index of the circle
                                        that caused the minimum intersection for each line.
    """
    # N = number of lines, M = number of circles
    N = p1s.shape[0]
    M = centers.shape[0]

    if M == 0:
        return (xp.full(N, xp.inf), xp.full(N, -1, dtype=xp.int32))

    # Line segment vectors (direction)
    d = p2s - p1s  # Shape (N, 2)

    # Vector from each line start to each circle center
    # Use broadcasting to create a matrix of vectors of shape (N, M, 2)
    f = p1s[:, xp.newaxis, :] - centers[xp.newaxis, :, :]

    # Coefficients of the quadratic equation at^2 + bt + c = 0, derived from
    # ||P1 + t*d - C||^2 = R^2
    # All coefficients will have shape (N, M)
    a = xp.sum(d * d, axis=1)[:, xp.newaxis]
    b = 2 * xp.sum(d[:, xp.newaxis, :] * f, axis=2)
    c = xp.sum(f * f, axis=2) - radii[xp.newaxis, :]**2

    # Calculate discriminant (b^2 - 4ac)
    discriminant = b**2 - 4 * a * c

    # Create a mask for valid intersections (discriminant >= 0)
    # Using 'where' is often safer than direct masking for assignment
    discriminant = xp.where(discriminant >= 0, discriminant, xp.nan)

    # Calculate the two potential solutions for t
    sqrt_discriminant = xp.sqrt(discriminant)
    a_broadcast = xp.broadcast_to(a, (N, M)) # Ensure 'a' has the same shape as others

    # Suppress "invalid value" warnings for divisions by zero or NaN, as we handle them
    with xp.errstate(divide='ignore', invalid='ignore'):
        t1 = (-b - sqrt_discriminant) / (2 * a_broadcast)
        t2 = (-b + sqrt_discriminant) / (2 * a_broadcast)

    # Invalidate t-values that are not on the line segment [0, 1]
    t1 = xp.where((t1 >= 0) & (t1 <= 1), t1, xp.nan)
    t2 = xp.where((t2 >= 0) & (t2 <= 1), t2, xp.nan)

    # Stack the two t-solutions to easily find the minimum valid t for each line-circle pair
    all_t = xp.stack([t1, t2], axis=0)

    # nanmin ignores NaNs, giving us the smallest valid t (or NaN if no valid t)
    min_t = xp.nanmin(all_t, axis=0) # Shape (N, M)

    # Calculate intersection distances from the start of the line
    line_lengths = xp.linalg.norm(d, axis=1)[:, xp.newaxis] # Shape (N, 1)
    distances = min_t * line_lengths # Shape (N, M)

    # Find the minimum distance for each line across all circles
    min_distances_per_line = xp.nanmin(distances, axis=1) # Shape (N,)

    # Get the index of the circle that caused the minimum distance.
    # We must handle the case where a line has no intersections (all-NaN row).
    min_indices_per_line = xp.full(N, -1, dtype=xp.int32)

    # Create a mask for rows that have at least one valid distance
    any_hit_mask = ~xp.all(xp.isnan(distances), axis=1)

    # Only run nanargmin on the rows that actually had a hit
    if xp.any(any_hit_mask):
        min_indices_per_line[any_hit_mask] = xp.nanargmin(distances[any_hit_mask], axis=1)

    # Where the min distance is still NaN, it means no intersection occurred.
    # The index for these is already correctly set to -1.
    min_distances_per_line[xp.isnan(min_distances_per_line)] = xp.inf

    return min_distances_per_line, min_indices_per_line


import pygame
import numpy as np

def iterative_line_circle_intersection(p1, p2, circle_center, radius):
    """
    Calculates the intersection of a line segment and a circle. Non-vectorized.
    Returns the distance from p1 to the closest intersection point, or None.
    """
    p1 = pygame.Vector2(p1)
    p2 = pygame.Vector2(p2)
    circle_center = pygame.Vector2(circle_center)

    d = p2 - p1
    f = p1 - circle_center

    a = d.dot(d)
    if a == 0: # Line segment has zero length
        return None

    b = 2 * f.dot(d)
    c = f.dot(f) - radius * radius

    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    else:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)

        # Check if the intersection points are on the line segment
        if 0 <= t1 <= 1:
            # t1 is an intersection on the segment
            return p1.distance_to(p1 + t1 * d)
        # We don't care about t2 if t1 is valid, as t1 will be closer
        return None
