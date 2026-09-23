"""
geo_qa.metrics
--------------
Compute area and extent figures directly from geometry — never from a
stored attribute column. Any "area" field baked into a shapefile at
creation time is a frozen snapshot: if a polygon is edited afterward
without re-running the calculation, the stored value goes stale with no
warning. `geometry.area` is always live and correct for the current
coordinates, so it's the only figure worth trusting.
"""

import geopandas as gpd
from . import validation


def compute_areas(gdf: gpd.GeoDataFrame, out_col: str = "area_m2",
                   skip_validity_check: bool = False) -> gpd.GeoDataFrame:
    """
    Add a freshly computed area column (in the GeoDataFrame's CRS units).

    Requires `gdf` to already be clean — raises if it isn't, rather than
    silently computing a wrong number. Reuses validation.diagnose_geometries()
    rather than re-checking only .is_valid here, so this always tests for
    everything validation.py knows to test for (null, empty, NaN bounds,
    and topological invalidity) — not a narrower, separately-maintained
    copy of that logic that could drift out of sync with it.

    Set skip_validity_check=True only if you've already validated
    separately and want to skip re-checking for performance on a very
    large dataset.
    """
    if not skip_validity_check:
        suspects = validation.diagnose_geometries(gdf)
        if suspects.any():
            raise ValueError(
                f"{suspects.sum()} rows failed geometry validation "
                f"(see counts above). Run validation.repair_and_clean() first."
            )

    gdf = gdf.copy()
    gdf[out_col] = gdf.geometry.area
    return gdf


def dataset_extent_summary(gdf: gpd.GeoDataFrame) -> dict:
    """
    Summarise how much ground a labeled dataset actually covers.

    Reports three different numbers because they answer different questions:
      - bounding_box_km2: the outer rectangle — usually an overestimate.
      - convex_hull_km2:  smallest convex shape enclosing all labels — better,
                           but can still overstate coverage if labels are
                           scattered in separate clusters with gaps between
                           them (a hull cannot have concave dents).
      - total_label_area_km2 / hull ratio: rough density — how much of the
        hull is actually built-up vs open space, paths, gaps.
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox_km2 = ((maxx - minx) * (maxy - miny)) / 1e6

    hull = gdf.geometry.union_all().convex_hull
    hull_km2 = hull.area / 1e6

    total_label_area_km2 = gdf.geometry.area.sum() / 1e6

    summary = {
        "n_structures": len(gdf),
        "bounding_box_km2": round(bbox_km2, 3),
        "convex_hull_km2": round(hull_km2, 3),
        "total_label_area_km2": round(total_label_area_km2, 4),
        "density_within_hull": round(total_label_area_km2 / hull_km2, 3) if hull_km2 else None,
    }
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    return summary
