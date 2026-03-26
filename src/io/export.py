"""Export cluster analysis results to CSV."""

import numpy as np
import pandas as pd
from pyrocko import moment_tensor as pmt

from ..config import ProcessingConfig
from ..geometry.coordinates import (
    strike_dip_to_normal,
    angle_between_planes,
    angle_vector_to_plane,
    angular_difference,
)
from ..geometry.pca import (
    cluster_to_xyz_enu,
    analyze_cluster_geometry,
    bootstrap_plane_uncertainty,
)
from ..geometry.hull import calculate_convex_hull_volume
from ..moment_tensor.analysis import (
    align_mt_planes_to_reference,
    mt_cosine_distance,
)


# -- Column ordering ------------------------------------------------

_COLUMNS_BASE = [
    "cluster_id",
    "num_events",
    "geometry_type",
    "linearity",
    "planarity",
    "volume_1e6m3",
    "pca_strike_deg",
    "pca_dip_deg",
    "pca_line_azimuth_deg",
    "pca_line_plunge_deg",
    "pca_line_unit_north",
    "pca_line_unit_east",
    "pca_line_unit_down",
]

_COLUMNS_BOOTSTRAP = [
    "pca_strike_std_deg",
    "pca_dip_std_deg",
    "pca_normal_std_deg",
    "pca_line_azimuth_std_deg",
    "pca_line_plunge_std_deg",
    "pca_line_direction_std_deg",
    "pca_line_unit_north_std",
    "pca_line_unit_east_std",
    "pca_line_unit_down_std",
]

_COLUMNS_MT = [
    "mt_plane1_strike_deg",
    "mt_plane1_dip_deg",
    "mt_plane1_rake_deg",
    "mt_plane1_strike_variation_deg",
    "mt_plane1_dip_variation_deg",
    "mt_plane2_strike_deg",
    "mt_plane2_dip_deg",
    "mt_plane2_rake_deg",
    "mt_plane2_strike_variation_deg",
    "mt_plane2_dip_variation_deg",
    "pca_plane_to_mt_plane1_angle_deg",
    "pca_plane_to_mt_plane2_angle_deg",
    "pca_line_to_mt_plane1_angle_deg",
    "pca_line_to_mt_plane2_angle_deg",
    "best_mt_plane",
    "mt_best_match_angle_deg",
    "mt_cos_representative_plane1_angle_deg",
    "linear_line_to_cos_rep_mt_plane_angle_deg",
    "linear_line_to_best_mt_plane_angle_deg",
    "num_mts",
    "mt_pairwise_kagan_mean_deg",
    "mt_pairwise_kagan_std_deg",
    "mt_to_cos_representative_kagan_mean_deg",
    "mt_to_cos_representative_kagan_std_deg",
]


# -- Internal helpers ------------------------------------------------

def _build_pca_record(geom, cid, vol, cfg):
    """Build the base PCA fields for one cluster."""
    d = cfg.decimals
    is_linear = geom["linearity"] > geom["planarity"]

    elong_ned = (
        geom["elongation_ned"]
        / np.linalg.norm(geom["elongation_ned"])
    )
    en, ee, ed = float(elong_ned[0]), float(elong_ned[1]), float(elong_ned[2])

    rec = {
        "cluster_id": cid,
        "num_events": geom["num_events"],
        "linearity": round(geom["linearity"], d),
        "planarity": round(geom["planarity"], d),
        "geometry_type": "linear" if is_linear else "planar",
        "volume_1e6m3": vol,
        "pca_strike_deg": (
            None if is_linear
            else round(geom["strike"], d)
        ),
        "pca_dip_deg": (
            None if is_linear
            else round(geom["dip"], d)
        ),
        "pca_line_azimuth_deg": (
            round(geom["elong_az"], d) if is_linear else None
        ),
        "pca_line_plunge_deg": (
            round(geom["elong_plunge"], d) if is_linear else None
        ),
        "pca_line_unit_north": (
            round(en, d) if is_linear else None
        ),
        "pca_line_unit_east": (
            round(ee, d) if is_linear else None
        ),
        "pca_line_unit_down": (
            round(ed, d) if is_linear else None
        ),
    }
    return rec, is_linear, elong_ned


def _add_mt_fields(rec, cluster, geom, is_linear, elong_ned, cfg):
    """Compare PCA geometry to the representative MT."""
    d = cfg.decimals
    mts = [
        ev.moment_tensor for ev in cluster
        if hasattr(ev, "moment_tensor") and ev.moment_tensor
    ]
    if not mts:
        rec.update({c: None for c in _COLUMNS_MT})
        return

    mt_objs = [pmt.as_mt(m) for m in mts]

    # --- representative MT (min cosine-distance sum) ---
    if len(mt_objs) == 1:
        best_mt = mt_objs[0]
    else:
        total_cos = [
            sum(mt_cosine_distance(a, b) for b in mt_objs)
            for a in mt_objs
        ]
        best_mt = mt_objs[int(np.argmin(total_cos))]

    planes = pmt.as_mt(best_mt).both_strike_dip_rake()

    # --- Kagan angle statistics ---
    kagan_mean, kagan_std = None, None
    if len(mt_objs) > 1:
        kagan = [
            pmt.kagan_angle(mt_objs[i], mt_objs[j])
            for i in range(len(mt_objs))
            for j in range(i + 1, len(mt_objs))
        ]
        if kagan:
            kagan_mean = round(np.mean(kagan), d)
            kagan_std = round(np.std(kagan), d)

    rep_kagan = [
        pmt.kagan_angle(best_mt, m) for m in mt_objs
    ]
    rep_kagan_mean = round(np.mean(rep_kagan), d)
    rep_kagan_std = round(np.std(rep_kagan), d)

    # --- per-plane variation across cluster MTs ---
    aligned = [
        align_mt_planes_to_reference(
            planes, m.both_strike_dip_rake(),
        )
        for m in mt_objs
    ]
    p1_ds = [
        angular_difference(a[0][0], planes[0][0], 180.0)
        for a in aligned
    ]
    p1_dd = [abs(a[0][1] - planes[0][1]) for a in aligned]
    p2_ds = [
        angular_difference(a[1][0], planes[1][0], 180.0)
        for a in aligned
    ]
    p2_dd = [abs(a[1][1] - planes[1][1]) for a in aligned]

    s1, d1, r1 = planes[0]
    s2, d2, r2 = planes[1]
    n_mt1 = strike_dip_to_normal(s1, d1)
    n_mt2 = strike_dip_to_normal(s2, d2)

    # --- PCA vs MT comparison ---
    a_mt1, a_mt2 = None, None
    e_mt1, e_mt2 = None, None
    best_plane, min_ang, ang_rep = None, None, None

    if is_linear:
        e_mt1 = round(angle_vector_to_plane(elong_ned, n_mt1), d)
        e_mt2 = round(angle_vector_to_plane(elong_ned, n_mt2), d)
        best_plane = 1 if e_mt1 < e_mt2 else 2
        min_ang = min(e_mt1, e_mt2)
        ang_rep = e_mt1
    else:
        n_pca = strike_dip_to_normal(
            geom["strike"], geom["dip"],
        )
        a_mt1 = round(angle_between_planes(n_pca, n_mt1), d)
        a_mt2 = round(angle_between_planes(n_pca, n_mt2), d)
        best_plane = 1 if a_mt1 < a_mt2 else 2
        min_ang = min(a_mt1, a_mt2)
        ang_rep = a_mt1

    rec.update({
        "mt_plane1_strike_deg": round(s1, d),
        "mt_plane1_dip_deg": round(d1, d),
        "mt_plane1_rake_deg": round(r1, d),
        "mt_plane1_strike_variation_deg": round(np.mean(p1_ds), d),
        "mt_plane1_dip_variation_deg": round(np.mean(p1_dd), d),
        "mt_plane2_strike_deg": round(s2, d),
        "mt_plane2_dip_deg": round(d2, d),
        "mt_plane2_rake_deg": round(r2, d),
        "mt_plane2_strike_variation_deg": round(np.mean(p2_ds), d),
        "mt_plane2_dip_variation_deg": round(np.mean(p2_dd), d),
        "pca_plane_to_mt_plane1_angle_deg": (
            a_mt1 if not is_linear else None
        ),
        "pca_plane_to_mt_plane2_angle_deg": (
            a_mt2 if not is_linear else None
        ),
        "pca_line_to_mt_plane1_angle_deg": (
            e_mt1 if is_linear else None
        ),
        "pca_line_to_mt_plane2_angle_deg": (
            e_mt2 if is_linear else None
        ),
        "best_mt_plane": best_plane,
        "mt_best_match_angle_deg": min_ang,
        "mt_cos_representative_plane1_angle_deg": ang_rep,
        "linear_line_to_cos_rep_mt_plane_angle_deg": (
            ang_rep if is_linear else None
        ),
        "linear_line_to_best_mt_plane_angle_deg": (
            min_ang if is_linear else None
        ),
        "num_mts": len(mts),
        "mt_pairwise_kagan_mean_deg": kagan_mean,
        "mt_pairwise_kagan_std_deg": kagan_std,
        "mt_to_cos_representative_kagan_mean_deg": rep_kagan_mean,
        "mt_to_cos_representative_kagan_std_deg": rep_kagan_std,
    })


def _add_bootstrap_fields(rec, xyz, is_linear, cfg):
    """Add bootstrap uncertainty columns."""
    d = cfg.decimals
    bs = bootstrap_plane_uncertainty(
        xyz, n_bootstrap=cfg.bootstrap_samples,
    )
    if bs is None:
        rec.update({c: None for c in _COLUMNS_BOOTSTRAP})
        return

    def _r(val):
        if val is None or np.isnan(val):
            return None
        return round(val, d)

    rec.update({
        "pca_normal_std_deg": _r(bs["normal_std_deg"]),
        "pca_strike_std_deg": (
            _r(bs["strike_std"]) if not is_linear else None
        ),
        "pca_dip_std_deg": (
            _r(bs["dip_std"]) if not is_linear else None
        ),
        "pca_line_azimuth_std_deg": (
            _r(bs["elong_az_std"]) if is_linear else None
        ),
        "pca_line_plunge_std_deg": (
            _r(bs["elong_plunge_std"]) if is_linear else None
        ),
        "pca_line_direction_std_deg": (
            _r(bs["elong_std_deg"]) if is_linear else None
        ),
        "pca_line_unit_north_std": (
            _r(bs["elong_n_std"]) if is_linear else None
        ),
        "pca_line_unit_east_std": (
            _r(bs["elong_e_std"]) if is_linear else None
        ),
        "pca_line_unit_down_std": (
            _r(bs["elong_d_std"]) if is_linear else None
        ),
    })


# -- Public API ------------------------------------------------------

def process_clusters_to_csv(
    cluster_dict: dict,
    output_csv: str,
    cfg: ProcessingConfig | None = None,
) -> pd.DataFrame:
    """Analyse every cluster and write results to *output_csv*.

    Parameters
    ----------
    cluster_dict : dict
        ``{cluster_id: [event, ...]}``
    output_csv : str
        Destination path for the CSV file.
    cfg : ProcessingConfig, optional
        Processing options.  Uses defaults when ``None``.

    Returns
    -------
    pandas.DataFrame
        The same table that was written to disk.
    """
    if cfg is None:
        cfg = ProcessingConfig()

    d = cfg.decimals
    records = []

    for cid, cluster in cluster_dict.items():
        xyz = cluster_to_xyz_enu(cluster)

        # --- volume ---
        vol = 0.0
        if cfg.include_volume and len(xyz) >= 4:
            vol_m3, _ = calculate_convex_hull_volume(np.array(xyz))
            vol = round(vol_m3 / cfg.volume_unit_divisor, d)

        # --- PCA ---
        geom = analyze_cluster_geometry(xyz)
        if geom is None:
            rec = {
                "cluster_id": cid,
                "num_events": len(xyz),
                "volume_1e6m3": vol,
            }
            records.append(rec)
            continue

        rec, is_linear, elong_ned = _build_pca_record(
            geom, cid, vol, cfg,
        )

        # --- MT comparison ---
        if cfg.include_mt_comparison:
            _add_mt_fields(
                rec, cluster, geom, is_linear, elong_ned, cfg,
            )

        # --- bootstrap ---
        if cfg.bootstrap:
            _add_bootstrap_fields(rec, xyz, is_linear, cfg)

        records.append(rec)

    # --- assemble DataFrame with stable column order ---
    df = pd.DataFrame(records)

    ordered = list(_COLUMNS_BASE)
    if cfg.bootstrap:
        ordered += _COLUMNS_BOOTSTRAP
    if cfg.include_mt_comparison:
        ordered += _COLUMNS_MT

    for col in ordered:
        if col not in df.columns:
            df[col] = None
    trailing = [c for c in df.columns if c not in ordered]
    df = df[ordered + trailing]

    df.to_csv(output_csv, index=False)
    return df
