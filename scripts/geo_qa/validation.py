"""
geo_qa.validation
------------------
Load a vector label file and find geometries that would silently corrupt
downstream calculations (area sums, unions, intersections) without
necessarily failing `is_valid`.

Why this exists: `gdf.geometry.is_valid` only catches topologically invalid
shapes (self-intersections etc). It does NOT catch null geometries, empty
(zero-vertex) geometries, or geometries with NaN coordinates — all of which
make shapely's `unary_union` / `intersection` throw a silent RuntimeWarning
and quietly drop or corrupt results rather than erroring out. This module
checks for all of these explicitly, not just `is_valid`.
"""

import geopandas as gpd
import pandas as pd


def load_shapefile(path: str) -> gpd.GeoDataFrame:
    """Load a shapefile/GeoJSON. Row count/CRS are on the returned gdf
    (len(gdf), gdf.crs) — printing that is the caller's job."""
    return gpd.read_file(path)


def diagnose_geometries(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, dict]:
    """
    Check every geometry for the failure modes that break union/intersection
    silently. Returns (suspects_mask, counts): a boolean mask (True =
    suspect) usable to isolate/drop problem rows, plus a dict of tallies
    for the caller to report however it likes (see reporting.py).

    Checks, in order:
      1. Null geometry (missing entirely)
      2. Empty geometry (zero-area / zero-vertex placeholder)
      3. NaN in the geometry's bounding box (corrupted coordinates)
      4. Topologically invalid (self-intersecting etc), checked only on
         rows that passed 1-3, since is_valid() can itself error on those.
    """
    null_geom = gdf.geometry.isna()
    empty_geom = gdf.geometry.is_empty.fillna(False)

    bounds = gdf.geometry.bounds
    nan_bounds = bounds.isna().any(axis=1)

    safe_rows = ~null_geom & ~empty_geom & ~nan_bounds
    invalid_topology = pd.Series(False, index=gdf.index)
    invalid_topology.loc[safe_rows] = ~gdf.geometry.loc[safe_rows].is_valid

    suspects = null_geom | empty_geom | nan_bounds | invalid_topology

    counts = {
        "null_geometries": int(null_geom.sum()),
        "empty_geometries": int(empty_geom.sum()),
        "nan_bounding_box": int(nan_bounds.sum()),
        "topologically_invalid": int(invalid_topology.sum()),
        "total_suspect": int(suspects.sum()),
        "total_rows": len(gdf),
    }

    return suspects, counts


def split_clean_and_suspect(
    gdf: gpd.GeoDataFrame, suspects_mask: pd.Series
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split a GeoDataFrame into (clean, suspect) based on a suspects mask.
    Drops suspects rather than repairing them (e.g. via buffer(0)), since
    that can silently change shapes — so you always know exactly what was
    excluded."""
    clean = gdf[~suspects_mask].reset_index(drop=True)
    suspect = gdf[suspects_mask]
    return clean, suspect
