"""
analyze_camp.py
----------------
End-to-end example: given a structure-footprint shapefile, work out whether
it's clean and how much usable training data it actually represents.

Usage:
    python analyze_camp.py path/to/footprints.shp --tile-size 256 --resolution 0.3 \
        --camp-outlines path/to/outlines.kml --camps "Camp 8E" "Camp 8W" "Camp 9"

Edit the SHAPEFILE_PATH default below, or pass a path on the command line.
"""

import argparse
import warnings

import geopandas as gpd
import matplotlib.pyplot as plot

import geo_qa


def main(shapefile_path: str, tile_size_px: int, resolution_m: float,
          min_coverage: float, out_prefix: str, camp_outlines_path: str, camps: list[str]):
    # spurious in shapely 2.0.1: inputs and results are valid
    warnings.filterwarnings("ignore", message="invalid value encountered in")

    print("\n=== 1. Load & validate geometries ===")
    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(gdf)} rows from {shapefile_path}")
    print(f"CRS: {gdf.crs}")
    # create a boolean series indicating teh suspects 
    suspect_mask_s, counts_dict = geo_qa.diagnose_geometries(gdf)
    #  use the series to output 2 gdf's (good and bad)
    clean_gdf = gdf[~suspect_mask_s].reset_index(drop=True)
    suspect_gdf = gdf[suspect_mask_s]
    for k, v in counts_dict.items():
        print(f"{k:>22}: {v}")
    print(f"Kept {len(clean_gdf)} clean rows, isolated {len(suspect_gdf)} suspect rows.")
    suspect_gdf.to_file(f"{out_prefix}_suspects.geojson")

    print("\n=== 2. Compute areas fresh from geometry (ignore stored fields) ===")
    clean_gdf["area_m2"] = clean_gdf.geometry.area

    print("\n=== 3. Dataset extent summary ===")
    # union first: labels overlap, so summing per-label areas double-counts
    label_union = clean_gdf.geometry.union_all()
    minx, miny, maxx, maxy = clean_gdf.total_bounds
    bbox_km2 = (maxx - minx) * (maxy - miny) / 1e6
    hull_km2 = label_union.convex_hull.area / 1e6
    label_km2 = label_union.area / 1e6
    print(f"structures: {len(clean_gdf)} | bbox: {bbox_km2:.3f} km² | convex hull: {hull_km2:.3f} km² | "
          f"labelled: {label_km2:.4f} km² | density within hull: {label_km2 / hull_km2:.3f}")

    print(f"\n=== 4. Tile grid @ {tile_size_px}px / {resolution_m}m/px, inside surveyed camps ===")
    camps_gdf = gpd.read_file(camp_outlines_path).to_crs(clean_gdf.crs)
    camps_gdf = camps_gdf[camps_gdf["Name"].isin(camps)]
    aoi_poly = camps_gdf.geometry.union_all()
    tile_gdf = geo_qa.generate_tile_grid(camps_gdf, tile_size_px, resolution_m)
    tile_gdf = tile_gdf[tile_gdf.within(aoi_poly)].copy()
    print(f"{len(tile_gdf)} tiles fall within {', '.join(camps)}")
    tile_gdf["label_coverage"] = tile_gdf.geometry.intersection(label_union).area / tile_gdf.area
    usable_gdf = tile_gdf[tile_gdf["label_coverage"] > min_coverage]
    print(f"{len(usable_gdf)} usable tiles out of {len(tile_gdf)} "
          f"(threshold: {min_coverage:.0%} coverage)")

    print("\n=== 5. Diagnostic outputs ===")
    fig, ax = plot.subplots()
    tile_gdf["label_coverage"].hist(bins=30, ax=ax)
    ax.set_xlabel("Fraction of tile covered by labels")
    ax.set_ylabel("Number of tiles")
    ax.set_title("Label coverage distribution")
    fig.savefig(f"{out_prefix}_coverage_histogram.png", dpi=120)
    plot.close(fig)
    clean_gdf.to_file(f"{out_prefix}_cleaned.geojson")
    usable_gdf.to_file(f"{out_prefix}_usable_tiles.geojson")

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
    parser.add_argument("--camp-outlines", default="kutupalong/Camp_09/CXB_New_camp_outline_KML_20180528.kml",
                         help="Camp outline file (KML/shp) with a 'Name' column")
    parser.add_argument("--camps", nargs="+", default=["Camp 8E", "Camp 8W", "Camp 9"],
                         help="Camp names whose outlines are fully surveyed")
    args = parser.parse_args()

    main(args.shapefile, args.tile_size, args.resolution, args.min_coverage, args.out_prefix,
         args.camp_outlines, args.camps)
