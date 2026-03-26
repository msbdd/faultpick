"""Moment tensor plane analysis and alignment utilities."""

import math

import numpy as np
from pyrocko import moment_tensor as pmt

from ..geometry.coordinates import (
    strike_dip_to_normal,
    angle_between_planes,
)


def mt_cosine_distance(mti, mtj) -> float:
    """
    From SeisCloud (https://git.pyrocko.org/cesca/seiscloud/)
    Normalised cosine distance between two moment tensors.

    After Willemann (1993) and Tape & Tape, normalised in R^9 so
    the inner product lies in [-1, +1].  Returns a distance in
    [0, 1] where 0 = identical and 1 = most dissimilar.

    Parameters
    ----------
    mti, mtj
        Pyrocko :class:`MomentTensor` objects (need ``.mnn``,
        ``.mee``, ``.mdd``, ``.mne``, ``.mnd``, ``.med``).
    """
    ni = math.sqrt(
        mti.mnn**2 + mti.mee**2 + mti.mdd**2
        + 2.0 * mti.mne**2
        + 2.0 * mti.mnd**2
        + 2.0 * mti.med**2
    )
    nj = math.sqrt(
        mtj.mnn**2 + mtj.mee**2 + mtj.mdd**2
        + 2.0 * mtj.mne**2
        + 2.0 * mtj.mnd**2
        + 2.0 * mtj.med**2
    )
    nc = ni * nj
    ip = (
        mti.mnn * mtj.mnn
        + mti.mee * mtj.mee
        + mti.mdd * mtj.mdd
        + 2.0 * mti.mne * mtj.mne
        + 2.0 * mti.mnd * mtj.mnd
        + 2.0 * mti.med * mtj.med
    ) / nc
    ip = max(-1.0, min(1.0, ip))
    return 0.5 * (1.0 - ip)


def get_mt_planes(mt_thing) -> tuple:
    """Return both nodal planes ``((s1, d1, r1), (s2, d2, r2))`` via Pyrocko.

    *mt_thing* can be a Pyrocko :class:`MomentTensor`, a 3x3 matrix, or
    anything accepted by :func:`pyrocko.moment_tensor.as_mt`.
    """
    mt = pmt.as_mt(mt_thing)
    return mt.both_strike_dip_rake()


def align_mt_planes_to_reference(
    reference_planes: tuple,
    candidate_planes: tuple,
) -> tuple:
    """Re-order *candidate_planes* so that plane-1 best
    matches *reference_planes* plane-1."""
    ref_normals = [
        strike_dip_to_normal(p[0], p[1]) for p in reference_planes
    ]
    cand_normals = [
        strike_dip_to_normal(p[0], p[1]) for p in candidate_planes
    ]

    direct_cost = (
        angle_between_planes(ref_normals[0], cand_normals[0])
        + angle_between_planes(ref_normals[1], cand_normals[1])
    )
    swapped_cost = (
        angle_between_planes(ref_normals[0], cand_normals[1])
        + angle_between_planes(ref_normals[1], cand_normals[0])
    )

    if direct_cost <= swapped_cost:
        return candidate_planes
    return candidate_planes[1], candidate_planes[0]


def select_geometry_consistent_plane(
    mt_planes: tuple,
    pca_geometry: dict,
) -> int:
    """Select the MT nodal plane most consistent with PCA geometry.

    Parameters
    ----------
    mt_planes : tuple
        ``(plane1, plane2)`` from ``mt.both_strike_dip_rake()``.
    pca_geometry : dict
        Output of :func:`~faultpick.geometry.pca.analyze_cluster_geometry`
        (must contain ``'strike'`` and ``'dip'``).

    Returns
    -------
    int
        0 or 1 -- index of the better-matching nodal plane.
    """
    pca_normal = strike_dip_to_normal(
        pca_geometry["strike"], pca_geometry["dip"],
    )
    angles = [
        angle_between_planes(pca_normal, strike_dip_to_normal(p[0], p[1]))
        for p in mt_planes
    ]
    return int(np.argmin(angles))
