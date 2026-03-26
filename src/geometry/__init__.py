"""Geometry analysis: coordinates, PCA, ellipsoids, convex hulls."""

from .coordinates import (
    enu_to_ned,
    vector_to_strike_dip,
    strike_dip_to_normal,
    angle_between_planes,
    angular_difference,
    vector_to_az_plunge,
    az_plunge_to_ned,
    angle_vector_to_plane,
    enforce_upper_hemisphere,
    circular_std,
    axial_vector_std,
)
from .pca import (
    analyze_cluster_geometry,
    bootstrap_plane_uncertainty,
    cluster_to_xyz_enu,
)
from .hull import calculate_convex_hull_volume

__all__ = [
    "enu_to_ned",
    "vector_to_strike_dip",
    "strike_dip_to_normal",
    "angle_between_planes",
    "angular_difference",
    "vector_to_az_plunge",
    "az_plunge_to_ned",
    "angle_vector_to_plane",
    "enforce_upper_hemisphere",
    "circular_std",
    "axial_vector_std",
    "analyze_cluster_geometry",
    "bootstrap_plane_uncertainty",
    "cluster_to_xyz_enu",
    "calculate_convex_hull_volume",
]
