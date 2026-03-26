"""Outlier-removal methods for seismic event clusters."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def remove_outliers_lof(
    df: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Remove outliers per cluster using Local Outlier Factor.

    Parameters
    ----------
    df : DataFrame
        Must contain ``ClusterID``, ``East_km``, ``North_km``, ``Up_km``.
    n_neighbors : int
        LOF neighbourhood size.
    contamination : float
        Expected fraction of outliers.
    """
    cleaned = []

    for _, group in df.groupby("ClusterID"):
        if len(group) <= n_neighbors:
            cleaned.append(group)
            continue

        coords = group[["East_km", "North_km", "Up_km"]].values
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination,
        )
        labels = lof.fit_predict(coords)
        cleaned.append(group[labels == 1])

    return pd.concat(cleaned, ignore_index=True)


def remove_outliers_elliptic(
    df: pd.DataFrame,
    contamination: float = 0.1,
) -> pd.DataFrame:
    """Remove outliers per cluster using robust covariance.

    Parameters
    ----------
    df : DataFrame
        Must contain ``ClusterID``, ``East_km``, ``North_km``, ``Up_km``.
    contamination : float
        Expected fraction of outliers.
    """
    cleaned = []

    for _, group in df.groupby("ClusterID"):
        coords = group[["East_km", "North_km", "Up_km"]].values

        if len(coords) < 6:
            cleaned.append(group)
            continue

        ee = EllipticEnvelope(
            contamination=contamination, support_fraction=0.8,
        )

        try:
            labels = ee.fit_predict(coords)
            cleaned.append(group[labels == 1])
        except (ValueError, np.linalg.LinAlgError):
            cleaned.append(group)

    return pd.concat(cleaned, ignore_index=True)


def remove_outliers_dbscan(
    df: pd.DataFrame,
    eps: float = 1.0,
    min_samples: int = 5,
    scale: bool = True,
) -> pd.DataFrame:
    """Flag outliers per cluster using DBSCAN.

    Adds ``is_outlier`` (bool) and ``outlier_reason`` columns.  The caller
    can then drop outliers with ``df[~df.is_outlier]``.

    Parameters
    ----------
    df : DataFrame
        Must contain ``ClusterID``, ``East_km``, ``North_km``, ``Up_km``.
    eps : float
        DBSCAN neighbourhood radius.
    min_samples : int
        Minimum core-point neighbourhood size.
    scale : bool
        Whether to standardise coordinates before clustering.
    """
    df = df.copy()
    df["is_outlier"] = False
    df["outlier_reason"] = "clean"

    for _, group in df.groupby("ClusterID"):
        coords = group[["East_km", "North_km", "Up_km"]].values

        if len(coords) < min_samples:
            continue

        X = StandardScaler().fit_transform(coords) if scale else coords

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)

        noise_idx = group.index[labels == -1]
        df.loc[noise_idx, "is_outlier"] = True
        df.loc[noise_idx, "outlier_reason"] = "dbscan"

    return df
