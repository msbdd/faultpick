"""Test faultpick with synthetic clusters of known orientation.

Generates point clouds on planes with known strike/dip, verifies that
PCA analysis recovers the correct orientations, and runs the full
export pipeline on synthetic Pyrocko events.

"""

import numpy as np
import pytest

from faultpick.geometry.pca import (
    analyze_cluster_geometry,
    bootstrap_plane_uncertainty,
)
from faultpick.geometry.hull import calculate_convex_hull_volume
from faultpick.config import ProcessingConfig
from faultpick.io.export import process_clusters_to_csv


# ---------------------------------------------------------------------------
# Independent geometry for test data generation
#
# These functions replicate the strike/dip/azimuth/plunge conventions
# from textbook definitions (Aki & Richards) WITHOUT importing any
# faultpick code, so we can verify the main code against an independent
# implementation.
# ---------------------------------------------------------------------------

def _independent_strike_dip_to_normal_ned(
    strike_deg: float, dip_deg: float,
) -> np.ndarray:
    """Strike/dip -> upward-pointing unit normal in NED (independent impl).

    Convention (Aki & Richards, right-hand rule):
      - Dip direction = strike + 90 (clockwise)
      - Normal horizontal projection points toward dip direction
      - Normal points upward (Down component < 0 in NED)

    The dip azimuth is (strike + 90). The normal's horizontal projection
    points in that direction with magnitude sin(dip), and the vertical
    (Up) component is cos(dip).

    In NED: North = cos(dip_az)*sin(dip), East = sin(dip_az)*sin(dip),
            Down = -cos(dip)  (negative = pointing up).
    """
    dip_az_rad = np.radians(strike_deg + 90.0)
    d_rad = np.radians(dip_deg)

    north = np.sin(d_rad) * np.cos(dip_az_rad)
    east = np.sin(d_rad) * np.sin(dip_az_rad)
    down = -np.cos(d_rad)

    normal = np.array([north, east, down])
    return normal / np.linalg.norm(normal)


def _independent_angular_difference(
    a: float, b: float, period: float = 360.0,
) -> float:
    """Smallest absolute angular difference (independent impl)."""
    diff = (a - b + period / 2.0) % period - period / 2.0
    return abs(diff)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_planar_cluster_enu(
    strike_deg: float,
    dip_deg: float,
    n_points: int = 200,
    spread_in_plane: float = 1000.0,
    thickness: float = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an (N, 3) ENU point cloud on a plane with given strike/dip.

    Uses an independent normal-vector computation (not from faultpick).

    Parameters
    ----------
    strike_deg, dip_deg : float
        Target plane orientation.
    n_points : int
        Number of points.
    spread_in_plane : float
        Half-width of the uniform distribution along in-plane axes (metres).
    thickness : float
        Standard deviation of Gaussian noise normal to the plane (metres).
    rng : numpy Generator, optional
        For reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Normal in NED — independent implementation
    normal_ned = _independent_strike_dip_to_normal_ned(strike_deg, dip_deg)

    # Build an orthonormal basis for the plane (in NED)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal_ned, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])

    v1 = np.cross(normal_ned, arbitrary)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal_ned, v1)
    v2 /= np.linalg.norm(v2)

    # Generate in-plane coordinates + normal noise
    coords_in_plane = rng.uniform(
        -spread_in_plane, spread_in_plane, (n_points, 2),
    )
    noise_normal = rng.normal(0, thickness, n_points)

    points_ned = (
        coords_in_plane[:, 0:1] * v1
        + coords_in_plane[:, 1:2] * v2
        + noise_normal[:, None] * normal_ned
    )

    # Convert NED -> ENU: E=NED[1], N=NED[0], U=-NED[2]
    points_enu = np.column_stack([
        points_ned[:, 1],   # East
        points_ned[:, 0],   # North
        -points_ned[:, 2],  # Up
    ])

    return points_enu


def generate_linear_cluster_enu(
    azimuth_deg: float,
    plunge_deg: float,
    n_points: int = 200,
    length: float = 2000.0,
    width: float = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an elongated (linear) point cloud in ENU coordinates.

    Uses independent azimuth/plunge -> NED conversion (not from faultpick).

    Parameters
    ----------
    azimuth_deg : float
        Trend azimuth (degrees from North, clockwise).
    plunge_deg : float
        Plunge angle (degrees below horizontal).
    n_points : int
        Number of points.
    length : float
        Half-length along the elongation axis (metres).
    width : float
        Standard deviation of perpendicular scatter (metres).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    az_rad = np.radians(azimuth_deg)
    pl_rad = np.radians(plunge_deg)

    # Direction vector in NED — independent textbook formula
    # North = cos(plunge) * cos(azimuth)
    # East  = cos(plunge) * sin(azimuth)
    # Down  = sin(plunge)
    direction_ned = np.array([
        np.cos(pl_rad) * np.cos(az_rad),
        np.cos(pl_rad) * np.sin(az_rad),
        np.sin(pl_rad),
    ])

    # Two perpendicular vectors
    arbitrary = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction_ned, arbitrary)) > 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    p1 = np.cross(direction_ned, arbitrary)
    p1 /= np.linalg.norm(p1)
    p2 = np.cross(direction_ned, p1)
    p2 /= np.linalg.norm(p2)

    t = rng.uniform(-length, length, n_points)
    noise1 = rng.normal(0, width, n_points)
    noise2 = rng.normal(0, width, n_points)

    points_ned = (
        t[:, None] * direction_ned
        + noise1[:, None] * p1
        + noise2[:, None] * p2
    )

    # NED -> ENU
    points_enu = np.column_stack([
        points_ned[:, 1],   # East
        points_ned[:, 0],   # North
        -points_ned[:, 2],  # Up
    ])
    return points_enu


def make_pyrocko_events_on_plane(
    strike_deg: float,
    dip_deg: float,
    n_points: int = 200,
    cluster_id: int = 0,
    center_lat: float = 0.0,
    center_lon: float = 0.0,
    center_depth: float = 10000.0,
    spread_in_plane: float = 500.0,
    thickness: float = 30.0,
    rng: np.random.Generator | None = None,
) -> list:
    """Create synthetic Pyrocko Event objects scattered on a fault plane.

    Uses small displacements from a center point so that the lat/lon to
    local-ENU projection is accurate.
    """
    from pyrocko import model
    from pyrocko.orthodrome import ne_to_latlon

    if rng is None:
        rng = np.random.default_rng(42)

    enu = generate_planar_cluster_enu(
        strike_deg, dip_deg,
        n_points=n_points,
        spread_in_plane=spread_in_plane,
        thickness=thickness,
        rng=rng,
    )

    events = []
    for i in range(n_points):
        east, north, up = enu[i]
        lat, lon = ne_to_latlon(center_lat, center_lon, north, east)
        depth = center_depth - up  # depth = -up relative to center

        ev = model.Event(
            lat=float(lat),
            lon=float(lon),
            depth=float(depth),
        )
        ev.extras = {"cluster_number": cluster_id}
        events.append(ev)

    return events


# ---------------------------------------------------------------------------
# Tests: coordinate round-trips
# ---------------------------------------------------------------------------

class TestCoordinateRoundTrips:
    """Verify that main code's strike_dip_to_normal and vector_to_strike_dip
    are consistent with the independent test implementation."""

    @pytest.mark.parametrize("strike,dip", [
        (0, 45),
        (45, 60),
        (90, 30),
        (135, 75),
        (180, 45),
        (270, 20),
        (315, 85),
        (350, 10),
    ])
    def test_strike_dip_roundtrip(self, strike, dip):
        """Main code's forward+inverse should recover the input."""
        from faultpick.geometry.coordinates import (
            strike_dip_to_normal,
            vector_to_strike_dip,
        )
        normal = strike_dip_to_normal(strike, dip)
        s_out, d_out = vector_to_strike_dip(normal)

        assert _independent_angular_difference(strike, s_out, 360.0) < 1.0, (
            f"Strike mismatch: input={strike}, output={s_out}"
        )
        assert abs(dip - d_out) < 1.0, (
            f"Dip mismatch: input={dip}, output={d_out}"
        )

    @pytest.mark.parametrize("strike,dip", [
        (0, 45),
        (45, 60),
        (90, 30),
        (180, 45),
        (270, 20),
        (315, 85),
    ])
    def test_main_vs_independent_normal(self, strike, dip):
        """Main code's strike_dip_to_normal should agree with independent impl.

        Since normals are axial (n == -n), we compare via the acute angle
        between them.
        """
        from faultpick.geometry.coordinates import strike_dip_to_normal

        n_main = strike_dip_to_normal(strike, dip)
        n_indep = _independent_strike_dip_to_normal_ned(strike, dip)

        # Both should be unit vectors
        assert abs(np.linalg.norm(n_main) - 1.0) < 1e-10
        assert abs(np.linalg.norm(n_indep) - 1.0) < 1e-10

        # Angle between them (accounting for sign ambiguity)
        dot = np.clip(abs(np.dot(n_main, n_indep)), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))
        assert angle_deg < 1.0, (
            f"strike={strike}, dip={dip}: main vs independent normal "
            f"differ by {angle_deg:.2f} degrees.\n"
            f"  main:  {n_main}\n  indep: {n_indep}"
        )


# ---------------------------------------------------------------------------
# Tests: PCA on synthetic planar clusters
# ---------------------------------------------------------------------------

class TestPlanarClusters:
    """Test PCA recovery of known strike/dip on synthetic planar clusters."""

    TOLERANCE_DEG = 1

    @pytest.mark.parametrize("strike,dip", [
        (30, 60),    # NE-striking, moderately steep
        (150, 45),   # SE-striking, moderate dip
        (270, 30),   # W-striking, shallow dip
    ])
    def test_single_planar_cluster(self, strike, dip):
        rng = np.random.default_rng(12345)
        enu = generate_planar_cluster_enu(
            strike, dip,
            n_points=500,
            spread_in_plane=1000.0,
            thickness=20.0,
            rng=rng,
        )

        geom = analyze_cluster_geometry(enu)

        assert geom is not None
        assert geom["planarity"] > geom["linearity"], (
            f"Expected planar geometry, got linearity={geom['linearity']:.3f} "
            f"> planarity={geom['planarity']:.3f}"
        )

        strike_err = _independent_angular_difference(
            strike, geom["strike"], 360.0,
        )
        dip_err = abs(dip - geom["dip"])

        assert strike_err < self.TOLERANCE_DEG, (
            f"Strike: expected {strike}, got {geom['strike']:.1f} "
            f"(error {strike_err:.1f} > {self.TOLERANCE_DEG})"
        )
        assert dip_err < self.TOLERANCE_DEG, (
            f"Dip: expected {dip}, got {geom['dip']:.1f} "
            f"(error {dip_err:.1f} > {self.TOLERANCE_DEG})"
        )

    def test_three_distinct_clusters(self):
        """Generate 3 clusters with different orientations
        and verify recovery."""
        targets = [
            (30, 60),
            (150, 45),
            (270, 30),
        ]
        rng = np.random.default_rng(99)

        for strike, dip in targets:
            enu = generate_planar_cluster_enu(
                strike, dip,
                n_points=500,
                spread_in_plane=1000.0,
                thickness=20.0,
                rng=rng,
            )
            geom = analyze_cluster_geometry(enu)
            assert geom is not None

            strike_err = _independent_angular_difference(
                strike, geom["strike"], 360.0,
            )
            dip_err = abs(dip - geom["dip"])

            assert strike_err < self.TOLERANCE_DEG, (
                f"Cluster (strike={strike}, dip={dip}): "
                f"strike error {strike_err:.1f} > {self.TOLERANCE_DEG}"
            )
            assert dip_err < self.TOLERANCE_DEG, (
                f"Cluster (strike={strike}, dip={dip}): "
                f"dip error {dip_err:.1f} > {self.TOLERANCE_DEG}"
            )

    def test_bootstrap_uncertainty_small_for_tight_cluster(self):
        """A tight planar cluster should have small bootstrap uncertainties."""
        rng = np.random.default_rng(42)
        enu = generate_planar_cluster_enu(
            strike_deg=90, dip_deg=45,
            n_points=500,
            spread_in_plane=1000.0,
            thickness=10.0,
            rng=rng,
        )

        bs = bootstrap_plane_uncertainty(enu, n_bootstrap=200)
        assert bs is not None

        assert bs["strike_std"] < 5.0, (
            f"Strike std too large: {bs['strike_std']:.1f}"
        )
        assert bs["dip_std"] < 5.0, (
            f"Dip std too large: {bs['dip_std']:.1f}"
        )

    def test_near_horizontal_plane(self):
        """A nearly horizontal plane (dip~5) should be recovered accurately."""
        expected_dip = 5.0
        rng = np.random.default_rng(42)
        enu = generate_planar_cluster_enu(
            strike_deg=0, dip_deg=expected_dip,
            n_points=500,
            spread_in_plane=2000.0,
            thickness=10.0,
            rng=rng,
        )
        geom = analyze_cluster_geometry(enu)
        assert geom is not None
        dip_err = abs(expected_dip - geom["dip"])
        assert dip_err < self.TOLERANCE_DEG, (
            f"Dip: expected {expected_dip}, got {geom['dip']:.1f} "
            f"(error {dip_err:.1f} > {self.TOLERANCE_DEG})"
        )

    def test_near_vertical_plane(self):
        """A near-vertical plane (dip~85) should be recovered accurately."""
        expected_dip = 85.0
        rng = np.random.default_rng(42)
        enu = generate_planar_cluster_enu(
            strike_deg=45, dip_deg=expected_dip,
            n_points=500,
            spread_in_plane=1000.0,
            thickness=10.0,
            rng=rng,
        )
        geom = analyze_cluster_geometry(enu)
        assert geom is not None
        dip_err = abs(expected_dip - geom["dip"])
        assert dip_err < self.TOLERANCE_DEG, (
            f"Dip: expected {expected_dip}, got {geom['dip']:.1f} "
            f"(error {dip_err:.1f} > {self.TOLERANCE_DEG})"
        )


# ---------------------------------------------------------------------------
# Tests: PCA on synthetic linear clusters
# ---------------------------------------------------------------------------

class TestLinearClusters:
    """Test that PCA recovers elongation direction for linear clusters."""

    TOLERANCE_DEG = 5.0

    @pytest.mark.parametrize("azimuth,plunge", [
        (45, 30),
        (180, 10),
        (300, 60),
    ])
    def test_linear_cluster(self, azimuth, plunge):
        rng = np.random.default_rng(777)
        enu = generate_linear_cluster_enu(
            azimuth, plunge,
            n_points=500,
            length=2000.0,
            width=30.0,
            rng=rng,
        )

        geom = analyze_cluster_geometry(enu)
        assert geom is not None
        assert geom["linearity"] > geom["planarity"], (
            f"Expected linear geometry, got linearity={geom['linearity']:.3f}"
        )

        # Azimuth is ambiguous by 180 degrees (trend vs anti-trend)
        az_err = _independent_angular_difference(
            azimuth, geom["elong_az"], 180.0,
        )
        pl_err = abs(plunge - geom["elong_plunge"])

        assert az_err < self.TOLERANCE_DEG, (
            f"Azimuth: expected {azimuth}, got {geom['elong_az']:.1f} "
            f"(error {az_err:.1f})"
        )
        assert pl_err < self.TOLERANCE_DEG, (
            f"Plunge: expected {plunge}, got {geom['elong_plunge']:.1f} "
            f"(error {pl_err:.1f})"
        )


# ---------------------------------------------------------------------------
# Tests: Convex hull volume
# ---------------------------------------------------------------------------

class TestVolume:
    def test_planar_cluster_has_small_volume(self):
        """A thin planar cluster should have less volume than a thick one."""
        rng = np.random.default_rng(42)
        thin = generate_planar_cluster_enu(
            strike_deg=0, dip_deg=45,
            n_points=200,
            spread_in_plane=1000.0,
            thickness=10.0,
            rng=rng,
        )
        thick = generate_planar_cluster_enu(
            strike_deg=0, dip_deg=45,
            n_points=200,
            spread_in_plane=1000.0,
            thickness=500.0,
            rng=rng,
        )

        vol_thin, _ = calculate_convex_hull_volume(thin)
        vol_thick, _ = calculate_convex_hull_volume(thick)

        assert vol_thin < vol_thick, (
            f"Thin cluster volume ({vol_thin:.0f}) should be less than "
            f"thick cluster ({vol_thick:.0f})"
        )


# ---------------------------------------------------------------------------
# Tests: Full pipeline with synthetic Pyrocko events
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end test for synthetic Pyrocko clusters through CSV export."""

    TOLERANCE_DEG = 5.0

    def test_three_clusters_pipeline(self, tmp_path):
        """Generate 3 clusters with known orientations, run full pipeline."""
        targets = {
            0: (30, 60),
            1: (150, 45),
            2: (270, 30),
        }

        rng = np.random.default_rng(2024)
        cluster_dict = {}

        for cid, (strike, dip) in targets.items():
            events = make_pyrocko_events_on_plane(
                strike_deg=strike,
                dip_deg=dip,
                n_points=200,
                cluster_id=cid,
                center_lat=0.0 + cid * 0.1,  # slightly offset centers
                center_lon=0.0 + cid * 0.1,
                center_depth=10000.0,
                spread_in_plane=500.0,
                thickness=20.0,
                rng=rng,
            )
            cluster_dict[cid] = events

        output_csv = str(tmp_path / "results.csv")
        cfg = ProcessingConfig(
            bootstrap=True,
            bootstrap_samples=100,  # fewer for speed
            include_volume=True,
            include_mt_comparison=False,  # no MTs in synthetic data
            outlier_method=None,
            decimals=2,
        )

        df = process_clusters_to_csv(cluster_dict, output_csv, cfg)

        assert len(df) == 3
        assert set(df["cluster_id"]) == {0, 1, 2}

        for _, row in df.iterrows():
            cid = row["cluster_id"]
            expected_strike, expected_dip = targets[cid]

            assert row["geometry_type"] == "planar", (
                f"Cluster {cid}: expected planar, got {row['geometry_type']}"
            )

            strike_err = _independent_angular_difference(
                expected_strike, row["pca_strike_deg"], 360.0,
            )
            dip_err = abs(expected_dip - row["pca_dip_deg"])

            assert strike_err < self.TOLERANCE_DEG, (
                f"Cluster {cid}: strike error {strike_err:.1f}"
            )
            assert dip_err < self.TOLERANCE_DEG, (
                f"Cluster {cid}: dip error {dip_err:.1f}"
            )

            # Volume should be positive
            assert row["volume_1e6m3"] > 0, f"Cluster {cid}: zero volume"

            # Bootstrap std should be present and reasonable
            assert row["pca_strike_std_deg"] is not None
            assert row["pca_dip_std_deg"] is not None

        # Verify CSV was actually written
        import pandas as pd
        df_read = pd.read_csv(output_csv)
        assert len(df_read) == 3

    def test_too_few_events_handled(self, tmp_path):
        """Clusters with < 3 events should not crash the pipeline."""
        from pyrocko import model

        ev1 = model.Event(lat=0.0, lon=0.0, depth=10000.0)
        ev1.extras = {"cluster_number": 0}
        ev2 = model.Event(lat=0.0001, lon=0.0001, depth=10010.0)
        ev2.extras = {"cluster_number": 0}

        cluster_dict = {0: [ev1, ev2]}
        output_csv = str(tmp_path / "small.csv")

        cfg = ProcessingConfig(
            bootstrap=False,
            include_volume=False,
            include_mt_comparison=False,
            outlier_method=None,
        )
        df = process_clusters_to_csv(cluster_dict, output_csv, cfg)
        assert len(df) == 1
        assert df.iloc[0]["num_events"] == 2
