"""PCA-based cluster geometry analysis and bootstrap uncertainties."""

import numpy as np
from sklearn.decomposition import PCA
from pyrocko.orthodrome import latlon_to_ne_numpy

from .coordinates import (
    enu_to_ned,
    vector_to_strike_dip,
    vector_to_az_plunge,
    enforce_upper_hemisphere,
    circular_std,
    axial_vector_std,
)


def cluster_to_xyz_enu(cluster: list) -> np.ndarray:
    """Convert a list of Pyrocko events to an (N, 3) ENU array in metres.

    The local coordinate origin is the centroid of the cluster.
    """
    if not cluster:
        return np.empty((0, 3))

    lats = np.array([ev.lat for ev in cluster])
    lons = np.array([ev.lon for ev in cluster])
    deps = np.array([ev.depth for ev in cluster])

    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))

    north, east = latlon_to_ne_numpy(lat0, lon0, lats, lons)
    up = -deps

    return np.vstack([east, north, up]).T


def analyze_cluster_geometry(xyz_enu: np.ndarray) -> dict | None:
    """Run PCA on an ENU point cloud and return geometry metrics.

    Returns a dict with keys: num_events, linearity, planarity, eigenvalues,
    strike, dip, elong_az, elong_plunge, and raw direction vectors in both
    ENU and NED frames.  Returns ``None`` if fewer than 3 points.
    """
    if len(xyz_enu) < 3:
        return None

    pca = PCA(n_components=3)
    pca.fit(xyz_enu)

    evec1 = pca.components_[0]  # elongation direction
    evec3 = pca.components_[2]  # plane normal (least variance)
    lam1, lam2, lam3 = pca.explained_variance_

    if lam1 < 1e-12:
        return None

    linearity = (lam1 - lam2) / lam1
    planarity = (lam2 - lam3) / lam1

    normal_ned = enu_to_ned(evec3)
    elong_ned = enu_to_ned(evec1)

    strike, dip = vector_to_strike_dip(normal_ned)
    elong_az, elong_plunge = vector_to_az_plunge(elong_ned)

    return {
        "num_events": len(xyz_enu),
        "linearity": linearity,
        "planarity": planarity,
        "eigenvalues": (lam1, lam2, lam3),
        "strike": strike,
        "dip": dip,
        "elong_az": elong_az,
        "elong_plunge": elong_plunge,
        "normal_enu": evec3,
        "elongation_enu": evec1,
        "normal_ned": normal_ned,
        "elongation_ned": elong_ned,
    }


def bootstrap_plane_uncertainty(
    xyz_enu: np.ndarray,
    n_bootstrap: int = 1000,
) -> dict | None:
    """Estimate orientation uncertainties via bootstrap resampling.

    Returns a dict of standard-deviation estimates for strike, dip,
    elongation azimuth/plunge, and raw direction vectors.
    Returns ``None`` if fewer than 3 points.
    """
    if len(xyz_enu) < 3:
        return None

    normals = []
    elongations = []
    strike_list = []
    dip_list = []
    az_list = []
    plunge_list = []

    geom0 = analyze_cluster_geometry(xyz_enu)
    if geom0 is None:
        return None

    n_ref = enforce_upper_hemisphere(geom0["normal_ned"])
    e_ref = enforce_upper_hemisphere(geom0["elongation_ned"])

    for _ in range(n_bootstrap):
        idx = np.random.randint(0, len(xyz_enu), len(xyz_enu))
        geom = analyze_cluster_geometry(xyz_enu[idx])
        if geom is None:
            continue

        n_raw = geom["normal_ned"] / np.linalg.norm(geom["normal_ned"])
        e_raw = geom["elongation_ned"] / np.linalg.norm(geom["elongation_ned"])

        n = n_raw if np.dot(n_raw, n_ref) > 0 else -n_raw
        e = e_raw if np.dot(e_raw, e_ref) > 0 else -e_raw

        normals.append(n)
        elongations.append(e)

        s, d = vector_to_strike_dip(n)
        a, p = vector_to_az_plunge(e)

        strike_list.append(s % 360)
        dip_list.append(d)
        az_list.append(a % 360)
        plunge_list.append(p)

    return {
        "normal_std_deg": axial_vector_std(normals),
        "elong_std_deg": axial_vector_std(elongations),
        "strike_std": circular_std(strike_list, axial=True),
        "dip_std": np.nanstd(dip_list),
        "elong_az_std": circular_std(az_list, axial=True),
        "elong_plunge_std": np.nanstd(plunge_list),
        "elong_n_std": np.nanstd(np.asarray(elongations)[:, 0]),
        "elong_e_std": np.nanstd(np.asarray(elongations)[:, 1]),
        "elong_d_std": np.nanstd(np.asarray(elongations)[:, 2]),
    }
