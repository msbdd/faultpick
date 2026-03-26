"""Load seismic event catalogues and convert between
cluster representations."""

import warnings

import numpy as np
import pandas as pd
from pyrocko import model
from pyrocko.orthodrome import latlon_to_ne_numpy

try:
    import glasbey
except ImportError:  # optional at import time
    glasbey = None


def _parse_color(raw):
    """Try to turn *raw* into an (r, g, b) float tuple in [0, 1].

    Accepts hex strings like ``"#ff0000"`` or ``"ff0000"`` and
    RGB tuples/lists of ints (0-255) or floats (0-1).
    Returns ``None`` on failure.
    """
    if raw is None:
        return None

    if isinstance(raw, str):
        raw = raw.strip().lstrip("#")
        if len(raw) == 6:
            try:
                r = int(raw[0:2], 16) / 255.0
                g = int(raw[2:4], 16) / 255.0
                b = int(raw[4:6], 16) / 255.0
                return (r, g, b)
            except ValueError:
                return None
        return None

    if isinstance(raw, (list, tuple)) and len(raw) >= 3:
        r, g, b = raw[0], raw[1], raw[2]
        if all(isinstance(v, int) for v in (r, g, b)):
            return (r / 255.0, g / 255.0, b / 255.0)
        return (float(r), float(g), float(b))

    return None


def _resolve_cluster_color(cluster_id, events, color_fields):
    """Return the colour for a cluster from the events' extras.

    *color_fields* is a list of extras keys to try in order per event.
    If events carry different colours a warning is issued and
    ``None`` is returned so the caller can fall back to glasbey.
    """
    colors = set()
    for ev in events:
        for field in color_fields:
            c = _parse_color(ev.extras.get(field))
            if c is not None:
                colors.add(c)
                break

    if len(colors) == 1:
        return colors.pop()

    if len(colors) > 1:
        warnings.warn(
            f"Cluster {cluster_id}: events have {len(colors)} different "
            f"colours in extras['{color_fields}'] — falling back to glasbey.",
            stacklevel=2,
        )

    return None  # no colour or inconsistent


def load_clustered_events(
    event_file: str,
    cluster_field: str = "cluster_number",
    color_field: str | None = None,
) -> tuple[dict, dict, list]:
    """Load a Pyrocko event file and split into clustered / unclustered events.

    Parameters
    ----------
    event_file : str
        Path to a Pyrocko YAML event file.
    cluster_field : str
        Name of the ``extras`` field that holds the cluster number.
    color_field : str or None
        Name of the ``extras`` field that holds the event colour.
        When ``None`` (the default), both ``"color"`` and ``"colour"``
        are tried in that order.

    Returns
    -------
    clusters : dict
        ``{cluster_id: [event, ...]}``
    color_map : dict
        ``{cluster_id: (r, g, b)}``
    unclustered : list
        Events whose *cluster_field* equals -1.
    """
    events = model.load_events(event_file)

    clustered = [
        ev for ev in events
        if ev.extras.get(cluster_field, -1) != -1
    ]
    unclustered = [
        ev for ev in events
        if ev.extras.get(cluster_field, -1) == -1
    ]

    clusters: dict[int, list] = {}
    for ev in clustered:
        cid = int(ev.extras[cluster_field])
        clusters.setdefault(cid, []).append(ev)

    # --- resolve colours ---
    color_fields = (
        [color_field] if color_field is not None else ["color", "colour"]
    )
    color_map: dict[int, tuple] = {}
    needs_glasbey = []

    for cid, evts in sorted(clusters.items()):
        c = _resolve_cluster_color(cid, evts, color_fields)
        if c is not None:
            color_map[cid] = c
        else:
            needs_glasbey.append(cid)

    if needs_glasbey:
        if glasbey is not None:
            palette = glasbey.create_palette(
                len(needs_glasbey), as_hex=False,
            )
            for cid, color in zip(needs_glasbey, palette):
                color_map[cid] = color
        else:
            for cid in needs_glasbey:
                color_map[cid] = (0.5, 0.5, 0.5)

    return clusters, color_map, unclustered


def cluster_dict_to_df(cluster_dict: dict) -> pd.DataFrame:
    """Convert a cluster dict to a DataFrame with local ENU coordinates (km).

    A single reference point (centroid of all events) is used for the
    coordinate projection so that positions are comparable across clusters.

    Each row stores the original Pyrocko event object in the ``"event"``
    column so that the round-trip via :func:`df_to_cluster_dict` is lossless.
    """
    all_events = [
        (cid, ev)
        for cid, cluster in cluster_dict.items()
        for ev in cluster
    ]
    if not all_events:
        return pd.DataFrame(
            columns=["ClusterID", "event", "East_km", "North_km", "Up_km"],
        )

    all_lats = np.array([ev.lat for _, ev in all_events])
    all_lons = np.array([ev.lon for _, ev in all_events])
    all_deps = np.array([ev.depth for _, ev in all_events])

    lat0 = float(np.mean(all_lats))
    lon0 = float(np.mean(all_lons))

    north, east = latlon_to_ne_numpy(lat0, lon0, all_lats, all_lons)
    up = -all_deps

    rows = []
    for (cid, ev), e, n, u in zip(all_events, east, north, up):
        rows.append({
            "ClusterID": cid,
            "event": ev,
            "East_km": e / 1000.0,
            "North_km": n / 1000.0,
            "Up_km": u / 1000.0,
        })

    return pd.DataFrame(rows)


def df_to_cluster_dict(df: pd.DataFrame) -> dict:
    """Reconstruct a cluster dict from a DataFrame produced by
    :func:`cluster_dict_to_df`."""
    cleaned: dict[int, list] = {}
    for cid, group in df.groupby("ClusterID"):
        cleaned[cid] = list(group["event"])
    return cleaned
