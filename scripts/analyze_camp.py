"""
analyze_camp.py
----------------
End-to-end example: given a structure-footprint shapefile, work out whether
it's clean and how much usable training data it actually represents.

Usage:
    python analyze_camp.py path/to/footprints.shp --tile-size 256 --resolution 0.3

Edit the SHAPEFILE_PATH default below, or pass a path on the command line.
"""

import argparse
from geo_qa import validation, metrics, tiling, reporting


def main(shapefile_path: str, tile_size_px: int, resolution_m: float,
          min_coverage: float, out_prefix: str):

    print("\n=== 1. Load & validate geometries ===")
    gdf = validation.load_shapefile(shapefile_path)
    clean, suspect = validation.repair_and_clean(gdf)
    reporting.export_geojson(suspect, f"{out_prefix}_suspects.geojson")

    print("\n=== 2. Compute areas fresh from geometry (ignore stored fields) ===")
    clean = metrics.compute_areas(clean)

    print("\n=== 3. Dataset extent summary ===")
    metrics.dataset_extent_summary(clean)

    print(f"\n=== 4. Tile grid @ {tile_size_px}px / {resolution_m}m/px ===")
    tile_gdf = tiling.generate_tile_grid(clean, tile_size_px, resolution_m)
    tile_gdf = tiling.compute_tile_label_coverage(tile_gdf, clean)
    usable = tiling.filter_usable_tiles(tile_gdf, min_coverage)

    print("\n=== 5. Diagnostic outputs ===")
    reporting.plot_coverage_histogram(tile_gdf, f"{out_prefix}_coverage_histogram.png")
    reporting.export_geojson(clean, f"{out_prefix}_cleaned.geojson")
    reporting.export_geojson(usable, f"{out_prefix}_usable_tiles.geojson")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shapefile", help="Path to the structure-footprint shapefile")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size in pixels")
    parser.add_argument("--resolution", type=float, default=0.3, help="Metres per pixel")
    parser.add_argument("--min-coverage", type=float, default=0.05,
                         help="Minimum label coverage fraction for a tile to count as usable")
    parser.add_argument("--out-prefix", default="camp_analysis",
                         help="Prefix for output files")
    args = parser.parse_args()

    main(args.shapefile, args.tile_size, args.resolution, args.min_coverage, args.out_prefix)
