"""Coordinate transforms and angular geometry for seismological vectors.

All functions use the NED (North-East-Down) convention unless stated otherwise.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate system conversions
# ---------------------------------------------------------------------------

def enu_to_ned(v: np.ndarray) -> np.ndarray:
    """Convert an ENU (East-North-Up) vector to NED (North-East-Down)."""
    e, n, u = v
    return np.array([n, e, -u])


# ---------------------------------------------------------------------------
# Strike / dip / normal conversions
# ---------------------------------------------------------------------------

def vector_to_strike_dip(normal_ned: np.ndarray) -> tuple[float, float]:
    """Convert a normal vector (NED) to strike and dip.

    Convention (right-hand rule):
      - Strike: clockwise from North (0-360).
      - Dip: angle from horizontal (0-90).
      - Dip direction: 90 clockwise from strike.
      - Upward normal's horizontal projection points toward the dip direction.
    """
    n = normal_ned / np.linalg.norm(normal_ned)
    nx, ny, nz = n

    # Force normal to point upward (nz < 0 in NED means pointing up)
    if nz > 0:
        nx, ny, nz = -nx, -ny, -nz

    dip = np.degrees(np.arccos(-nz))

    if dip < 0.1:
        return 0.0, 0.0

    dip_az = np.degrees(np.arctan2(ny, nx)) % 360
    strike = (dip_az - 90.0) % 360

    return strike, dip


def strike_dip_to_normal(strike: float, dip: float) -> np.ndarray:
    """Convert strike/dip (degrees) to a unit normal vector (NED).

    Returns an upward-pointing normal whose horizontal projection points
    toward the dip direction (Aki & Richards convention).
    """
    s_rad = np.radians(strike)
    d_rad = np.radians(dip)

    normal = np.array([
        -np.sin(d_rad) * np.sin(s_rad),  # North
        np.sin(d_rad) * np.cos(s_rad),   # East
        -np.cos(d_rad),                   # Down (negative = up)
    ])

    return normal / np.linalg.norm(normal)


# ---------------------------------------------------------------------------
# Azimuth / plunge conversions
# ---------------------------------------------------------------------------

def vector_to_az_plunge(v_ned: np.ndarray) -> tuple[float, float]:
    """Compute azimuth (from North, clockwise) and plunge (downward positive)
    from a direction vector in NED coordinates."""
    v = v_ned / np.linalg.norm(v_ned)
    nx, ny, nz = v

    if nz < 0:
        nx, ny, nz = -nx, -ny, -nz

    az = np.degrees(np.arctan2(ny, nx)) % 360
    plunge = np.degrees(np.arcsin(nz))

    return az, plunge


def az_plunge_to_ned(azimuth: float, plunge: float) -> np.ndarray:
    """Convert azimuth and plunge (degrees) to a unit vector in NED."""
    az_rad = np.radians(azimuth)
    pl_rad = np.radians(plunge)

    n = np.cos(pl_rad) * np.cos(az_rad)
    e = np.cos(pl_rad) * np.sin(az_rad)
    d = np.sin(pl_rad)

    return np.array([n, e, d])


# ---------------------------------------------------------------------------
# Angular geometry
# ---------------------------------------------------------------------------

def angle_between_planes(n1: np.ndarray, n2: np.ndarray) -> float:
    """Acute angle (0-90 deg) between two planes given their normal vectors."""
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    dot = np.clip(np.abs(np.dot(n1, n2)), 0.0, 1.0)
    return np.degrees(np.arccos(dot))


def angle_vector_to_plane(
    vector_ned: np.ndarray,
    plane_normal_ned: np.ndarray,
) -> float:
    """Angle between a vector and a plane.

    Returns 0 when in-plane, 90 when perpendicular.
    """
    v = vector_ned / np.linalg.norm(vector_ned)
    n = plane_normal_ned / np.linalg.norm(plane_normal_ned)
    dot = np.clip(np.abs(np.dot(v, n)), 0.0, 1.0)
    return np.degrees(np.arcsin(dot))


def angular_difference(
    angle_a: float,
    angle_b: float,
    period: float = 360.0,
) -> float:
    """Smallest absolute angular difference for a circular variable."""
    diff = (angle_a - angle_b + period / 2.0) % period - period / 2.0
    return abs(diff)


# ---------------------------------------------------------------------------
# Vector statistics
# ---------------------------------------------------------------------------

def enforce_upper_hemisphere(v: np.ndarray) -> np.ndarray:
    """Force vector into a consistent hemisphere (z >= 0)."""
    v = v / np.linalg.norm(v)
    if v[2] < 0:
        v = -v
    return v


def circular_std(angles_deg: np.ndarray, axial: bool = False) -> float:
    """Circular standard deviation.

    Parameters
    ----------
    angles_deg : array-like
        Angles in degrees.
    axial : bool
        If True, treats 0 and 180 as identical (for strikes / trends).
    """
    angles = np.asarray(angles_deg)

    if axial:
        angles = 2 * angles

    angles_rad = np.deg2rad(angles)
    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))

    R = np.sqrt(sin_mean**2 + cos_mean**2)

    if R < 1e-8:
        return np.nan

    circ_std_rad = np.sqrt(-2 * np.log(R))
    std_deg = np.rad2deg(circ_std_rad)

    if axial:
        return std_deg / 2.0
    return std_deg


def axial_vector_std(vectors: list | np.ndarray) -> float:
    """RMS angular deviation (degrees) for axial vectors (v == -v)."""
    V = np.asarray(vectors)
    V = V / np.linalg.norm(V, axis=1)[:, None]

    vmean = V.mean(axis=0)
    vmean /= np.linalg.norm(vmean)

    dots = np.clip(np.abs(V @ vmean), -1.0, 1.0)
    angles = np.arccos(dots)

    return np.degrees(np.sqrt(np.mean(angles**2)))
