"""Command-line interface for faultpick."""

import argparse
import sys

from .config import ProcessingConfig
from .io.loading import (
    load_clustered_events,
    cluster_dict_to_df,
    df_to_cluster_dict,
)
from .io.export import process_clusters_to_csv
from .clustering.outliers import (
    remove_outliers_dbscan,
    remove_outliers_lof,
    remove_outliers_elliptic,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="faultpick",
        description=(
            "Analyse seismic cluster geometry and compare "
            "with moment tensor solutions."
        ),
    )

    # --- required ---
    p.add_argument(
        "event_file",
        help="Pyrocko YAML event file with cluster info in extras.",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV path.",
    )

    # --- cluster field ---
    p.add_argument(
        "--cluster-field",
        default="cluster_number",
        help=(
            "Name of the extras field that holds the cluster id "
            "(default: cluster_number)."
        ),
    )
    p.add_argument(
        "--color-field",
        default="color",
        help=(
            "Name of the extras field that holds the event colour "
            "(default: tries both 'color' and 'colour')."
        ),
    )

    # --- outlier removal ---
    p.add_argument(
        "--outlier-method",
        choices=["dbscan", "lof", "elliptic", "none"],
        default="none",
        help="Outlier removal method (default: none).",
    )
    p.add_argument("--dbscan-eps", type=float, default=1.0)
    p.add_argument("--dbscan-min-samples", type=int, default=5)
    p.add_argument(
        "--no-dbscan-scale",
        action="store_true",
        help="Disable coordinate scaling before DBSCAN.",
    )
    p.add_argument("--lof-neighbors", type=int, default=20)
    p.add_argument("--lof-contamination", type=float, default=0.05)
    p.add_argument(
        "--elliptic-contamination", type=float, default=0.1,
    )

    # --- analysis options ---
    p.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap uncertainties (enabled by default).",
    )
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000).",
    )
    p.add_argument(
        "--no-volume",
        action="store_true",
        help="Skip convex-hull volume computation.",
    )
    p.add_argument(
        "--no-mt",
        action="store_true",
        help="Skip moment tensor comparison.",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Decimal places in output (default: 2).",
    )

    return p


def run(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)

    # --- build config ---
    outlier = (
        None if opts.outlier_method == "none"
        else opts.outlier_method
    )
    cfg = ProcessingConfig(
        bootstrap=not opts.no_bootstrap,
        bootstrap_samples=opts.bootstrap_samples,
        include_volume=not opts.no_volume,
        include_mt_comparison=not opts.no_mt,
        decimals=opts.decimals,
        outlier_method=outlier,
        dbscan_eps=opts.dbscan_eps,
        dbscan_min_samples=opts.dbscan_min_samples,
        dbscan_scale=not opts.no_dbscan_scale,
        lof_n_neighbors=opts.lof_neighbors,
        lof_contamination=opts.lof_contamination,
        elliptic_contamination=opts.elliptic_contamination,
    )

    # --- load ---
    print(f"Loading events from {opts.event_file} ...")
    clusters, color_map, unclustered = load_clustered_events(
        opts.event_file,
        cluster_field=opts.cluster_field,
        color_field=opts.color_field,
    )
    n_clusters = len(clusters)
    n_events = sum(len(v) for v in clusters.values())
    print(
        f"  {n_events} clustered events in "
        f"{n_clusters} clusters, "
        f"{len(unclustered)} unclustered."
    )

    # --- outlier removal ---
    if cfg.outlier_method is not None:
        print(f"Removing outliers ({cfg.outlier_method}) ...")
        df = cluster_dict_to_df(clusters)
        n_before = len(df)

        if cfg.outlier_method == "dbscan":
            df = remove_outliers_dbscan(
                df,
                eps=cfg.dbscan_eps,
                min_samples=cfg.dbscan_min_samples,
                scale=cfg.dbscan_scale,
            )
            df = df[~df.is_outlier].drop(
                columns=["is_outlier", "outlier_reason"],
            )
        elif cfg.outlier_method == "lof":
            df = remove_outliers_lof(
                df,
                n_neighbors=cfg.lof_n_neighbors,
                contamination=cfg.lof_contamination,
            )
        elif cfg.outlier_method == "elliptic":
            df = remove_outliers_elliptic(
                df,
                contamination=cfg.elliptic_contamination,
            )

        n_removed = n_before - len(df)
        print(f"  Removed {n_removed} outliers ({len(df)} remaining).")
        clusters = df_to_cluster_dict(df)

    # --- process & export ---
    print("Analysing cluster geometry ...")
    result = process_clusters_to_csv(clusters, opts.output, cfg=cfg)
    print(
        f"Done. {len(result)} clusters written to {opts.output}"
    )

    return result


def main():
    run()
    sys.exit(0)
