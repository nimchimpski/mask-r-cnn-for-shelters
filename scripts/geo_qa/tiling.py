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
    """
    tile_gdf = tile_gdf.copy()
    tile_gdf["label_coverage"] = tile_gdf.geometry.apply(
        lambda t: label_gdf.geometry.intersection(t).area.sum() / t.area
    )
    return tile_gdf


def filter_usable_tiles(
    tile_gdf: gpd.GeoDataFrame, min_coverage: float = 0.05
) -> gpd.GeoDataFrame:
    """Keep only tiles with at least `min_coverage` fraction labeled."""
    usable = tile_gdf[tile_gdf["label_coverage"] > min_coverage]
    print(f"{len(usable)} usable tiles out of {len(tile_gdf)} "
          f"(threshold: {min_coverage:.0%} coverage)")
    return usable
