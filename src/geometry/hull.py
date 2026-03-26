"""Convex hull volume calculation."""

import numpy as np
from scipy.spatial import ConvexHull


def calculate_convex_hull_volume(
    points: np.ndarray,
) -> tuple[float, ConvexHull | None]:
    """Calculate the volume of the 3-D convex hull of *points*.

    Parameters
    ----------
    points : (N, 3) array
        Cartesian coordinates.

    Returns
    -------
    volume : float
        Hull volume (same units cubed as *points*).  Returns 0 if < 4 points.
    hull : ConvexHull or None
    """
    if len(points) < 4:
        return 0.0, None
    hull = ConvexHull(points)
    return hull.volume, hull
