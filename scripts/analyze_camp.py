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
    print(f"Loaded {len(gdf)} rows from {shapefile_path}")
    print(f"CRS: {gdf.crs}")
    # create a boolean series indicating teh suspects 
    suspect_mask_s, counts_dict = validation.diagnose_geometries(gdf)
    #  use the series to output 2 gdf's (good and bad)
    clean_gdf, suspect_gdf = validation.split_clean_and_suspect(gdf, suspect_mask_s)
    #  count goods and bads
    counts_dict["kept"] = len(clean_gdf)
    counts_dict["isolated"] = len(suspect_gdf)
    # print summary
    reporting.print_diagnosis_summary(counts_dict)
    reporting.export_geojson(suspect_gdf, f"{out_prefix}_suspects.geojson")

    print("\n=== 2. Compute areas fresh from geometry (ignore stored fields) ===")
    clean_gdf = metrics.add_areas(clean_gdf)

    print("\n=== 3. Dataset extent summary ===")
    metrics.dataset_extent_summary(clean_gdf)

    print(f"\n=== 4. Tile grid @ {tile_size_px}px / {resolution_m}m/px ===")
    tile_gdf = tiling.generate_tile_grid(clean_gdf, tile_size_px, resolution_m)
    print(f"Generated {len(tile_gdf)} candidate tiles at {tile_size_px * resolution_m:.1f} m per side "
          f"({tile_size_px}px @ {resolution_m}m/px)")
    tile_gdf = tiling.add_tile_label_coverage(tile_gdf, clean_gdf)
    usable_gdf = tiling.filter_usable_tiles(tile_gdf, min_coverage)
    print(f"{len(usable_gdf)} usable tiles out of {len(tile_gdf)} "
          f"(threshold: {min_coverage:.0%} coverage)")

    print("\n=== 5. Diagnostic outputs ===")
    reporting.plot_coverage_histogram(tile_gdf, f"{out_prefix}_coverage_histogram.png")
    reporting.export_geojson(clean_gdf, f"{out_prefix}_cleaned.geojson")
    reporting.export_geojson(usable_gdf, f"{out_prefix}_usable_tiles.geojson")

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
