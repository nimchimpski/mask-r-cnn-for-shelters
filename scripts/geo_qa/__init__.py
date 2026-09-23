"""
geo_qa — small utilities for assessing whether a labeled structure-footprint
dataset is trustworthy and sufficient for training.

Typical flow (see analyze_camp.py for a full worked example):
    1. validation.load_shapefile()      -> raw GeoDataFrame
    2. validation.repair_and_clean()    -> clean vs suspect split
    3. metrics.dataset_extent_summary() -> bbox / hull / density
    4. tiling.generate_tile_grid()      -> candidate tiles
    5. tiling.compute_tile_label_coverage() + filter_usable_tiles()
    6. reporting.plot_coverage_histogram() / export_geojson()
"""

from . import validation, metrics, tiling, reporting

__all__ = ["validation", "metrics", "tiling", "reporting"]
