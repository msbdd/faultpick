"""Processing configuration for faultpick."""

from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Controls what gets computed and exported.

    Parameters
    ----------
    bootstrap : bool
        Compute bootstrap uncertainty estimates for PCA orientations.
    bootstrap_samples : int
        Number of bootstrap resamples.
    include_volume : bool
        Compute convex-hull volume for each cluster.
    include_mt_comparison : bool
        Compare PCA geometry against representative moment tensors.
    volume_unit_divisor : float
        Divisor applied to raw volume (m^3) before export.
        Default 1e6 gives units of 10^6 m^3.
    decimals : int
        Number of decimal places for rounding in the output CSV.

    Outlier removal
    ---------------
    outlier_method : str or None
        One of ``"dbscan"``, ``"lof"``, ``"elliptic"``, or ``None``
        to skip outlier removal.
    dbscan_eps : float
        DBSCAN neighbourhood radius (used when method is ``"dbscan"``).
    dbscan_min_samples : int
        DBSCAN minimum core-point size.
    dbscan_scale : bool
        Standardise coordinates before DBSCAN.
    lof_n_neighbors : int
        LOF neighbourhood size.
    lof_contamination : float
        LOF expected outlier fraction.
    elliptic_contamination : float
        Elliptic Envelope expected outlier fraction.
    """

    # --- PCA & bootstrap ---
    bootstrap: bool = True
    bootstrap_samples: int = 1000

    # --- Volume ---
    include_volume: bool = True
    volume_unit_divisor: float = 1e6

    # --- Moment tensor comparison ---
    include_mt_comparison: bool = True

    # --- Output formatting ---
    decimals: int = 2

    # --- Outlier removal ---
    outlier_method: str | None = "dbscan"
    dbscan_eps: float = 1.0
    dbscan_min_samples: int = 5
    dbscan_scale: bool = True
    lof_n_neighbors: int = 20
    lof_contamination: float = 0.05
    elliptic_contamination: float = 0.1
