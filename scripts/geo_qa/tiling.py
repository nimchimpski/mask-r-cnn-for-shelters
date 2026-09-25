"""
geo_qa.tiling
-------------
Grid an area of interest into candidate training tiles, then score each
tile by what fraction of it is actually covered by labeled structures.
This is the direct, ground-truth way to answer "how much usable training
data do I really have" — more reliable than bounding box or convex hull
alone, since it respects gaps and uneven density within the labeled area.
"""

import geopandas as gpd
import numpy as np
import warnings
import scipy
from shapely.geometry import box


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


def compute_tile_label_coverage(
    tile_gdf: gpd.GeoDataFrame, label_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    For each tile, compute the fraction of its area covered by label
    polygons. Uses a spatial index (via sjoin) implicitly through shapely's
    STRtree under the hood in recent geopandas, but for clarity/robustness
    this does a direct per-tile intersection — fine at ~a few hundred to a
    few thousand tiles; for much larger grids, pre-filter with gdf.sindex
    before intersecting.

    Suppresses shapely's "invalid value encountered in intersection"
    RuntimeWarning here specifically: it's a known, benign quirk that fires
    on empty-result intersections (tiles with zero overlap), not a sign of
    a wrong result — see shapely/shapely#1345. Since most tiles in a sparse
    grid legitimately have zero overlap, this fires constantly and is safe
    to silence rather than chase.
    """
    tile_gdf = tile_gdf.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
        tile_gdf["label_coverage"] = tile_gdf.geometry.apply(
            lambda t: label_gdf.geometry.intersection(t).area.sum() / t.area
        )
    return tile_gdf


def filter_tiles_within_aoi(
    tile_gdf: gpd.GeoDataFrame,
    aoi_gdf: gpd.GeoDataFrame,
    mode: str = "contains",
    min_overlap: float = 0.95,
) -> gpd.GeoDataFrame:
    """
    Keep only tiles that fall within a genuine surveyed-area boundary
    (e.g. the outline of a camp/region that was actually mapped) — NOT the
    bounding box of the labels themselves.


    aoi_gdf:  made by infer_surveyed_boundary() - a single (multi)polygon describing the actual surveyed boundary

    mode="contains" (default, safest): keep only tiles fully inside the
        boundary. Every pixel in a kept tile has a trustworthy label
        status. Fewer tiles, but no ambiguity.

    mode="overlap": keep tiles with at least `min_overlap` fraction of
        their area inside the boundary. 
    """
    aoi_union = aoi_gdf.geometry.union_all()

    if mode == "contains":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in contains")
            keep = tile_gdf.geometry.apply(lambda t: aoi_union.contains(t))
    elif mode == "overlap":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in intersection")
            overlap_frac = tile_gdf.geometry.apply(
                lambda t: t.intersection(aoi_union).area / t.area
            )
        keep = overlap_frac >= min_overlap
    else:
        raise ValueError(f"mode must be 'contains' or 'overlap', got {mode!r}")

    result = tile_gdf[keep].reset_index(drop=True)
    print(f"{len(result)} / {len(tile_gdf)} tiles fall within the AOI boundary "
          f"(mode={mode}" + (f", min_overlap={min_overlap:.0%}" if mode == "overlap" else "") + ")")
    if mode == "overlap" and len(result) > 0:
        print("Reminder: edge tiles here include unsurveyed area outside the AOI — "
              "mask that portion out of the training loss, do not treat it as background.")
    return result


def filter_usable_tiles(
    tile_gdf: gpd.GeoDataFrame, min_coverage: float = 0.05
) -> gpd.GeoDataFrame:
    """Keep only tiles with at least `min_coverage` fraction labeled."""
    usable = tile_gdf[tile_gdf["label_coverage"] > min_coverage]
    print(f"{len(usable)} usable tiles out of {len(tile_gdf)} "
          f"(threshold: {min_coverage:.0%} coverage)")
    return usable


def tile_class_balance(tile_gdf: gpd.GeoDataFrame) -> dict:
    """
    Summarize the label_coverage distribution across tiles, so a filtering
    threshold (or decision to not filter at all) can be chosen from the
    actual class balance instead of an arbitrary guess. Comparable across
    datasets of different sizes/densities since bins are tile counts.
    """
    coverage = tile_gdf["label_coverage"]
    return {
        "n_tiles": len(coverage),
        "mean_coverage": float(coverage.mean()) if len(coverage) else 0.0,
        "median_coverage": float(coverage.median()) if len(coverage) else 0.0,
        "bins": {
            "0%": int((coverage == 0).sum()),
            "0-5%": int(((coverage > 0) & (coverage <= 0.05)).sum()),
            "5-25%": int(((coverage > 0.05) & (coverage <= 0.25)).sum()),
            "25-50%": int(((coverage > 0.25) & (coverage <= 0.50)).sum()),
            "50-100%": int((coverage > 0.50).sum()),
        },
    }
