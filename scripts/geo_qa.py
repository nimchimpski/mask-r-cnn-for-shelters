"""
geo_qa — the few pieces with enough logic to be worth reusing.
Everything else is inlined in the scripts (see analyze_camp.py).
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def diagnose_geometries(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, dict]:
    """
    Find geometries that would silently corrupt area sums, unions and
    intersections. `is_valid` alone misses null, empty and NaN-coordinate
    geometries, which make shapely quietly drop or corrupt results.

    Returns (suspects_mask, counts): True = suspect, plus a dict of tallies.

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


def generate_tile_grid(
    gdf: gpd.GeoDataFrame, tile_size_px: int, resolution_m: float
) -> gpd.GeoDataFrame:
    """
    Build a regular grid of square tiles covering the bounding box of `gdf`.

    tile_size_px: tile side length in pixels (e.g. 256, matching your model
                  input size).
    resolution_m: ground sample distance in metres/pixel (e.g. 0.3 for PNEO,
                  0.5 for SkySat/Pléiades). tile_size_px * resolution_m gives
                  the tile's real-world side length in metres.
    """
    tile_size_m = tile_size_px * resolution_m
    minx, miny, maxx, maxy = gdf.total_bounds

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tiles.append(box(x, y, x + tile_size_m, y + tile_size_m))
            y += tile_size_m
        x += tile_size_m

    tile_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
    print(f"Generated {len(tile_gdf)} candidate tiles at {tile_size_m:.1f} m per side "
          f"({tile_size_px}px @ {resolution_m}m/px)")
    return tile_gdf
