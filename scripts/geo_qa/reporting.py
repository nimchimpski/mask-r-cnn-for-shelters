"""
geo_qa.reporting
----------------
Diagnostic outputs meant for a human to actually look at — a coverage
histogram (is coverage uniform or lopsided?) and exportable GeoJSON layers
(suspect geometries, cleaned data) to drag straight into QGIS.
"""

import geopandas as gpd
import matplotlib.pyplot as plt


def print_diagnosis_summary(counts: dict) -> None:
    """Print the counts dict returned by validation.diagnose_geometries()
    (optionally with "kept"/"isolated" added after split_clean_and_suspect())."""
    print(f"Null geometries:              {counts['null_geometries']}")
    print(f"Empty geometries:             {counts['empty_geometries']}")
    print(f"NaN bounding-box geometries:  {counts['nan_bounding_box']}")
    print(f"Topologically invalid:        {counts['topologically_invalid']}")
    print(f"Total suspect rows:           {counts['total_suspect']} / {counts['total_rows']}")
    if "kept" in counts:
        print(f"\nKept {counts['kept']} clean rows, isolated {counts['isolated']} suspect rows.")


def print_class_balance(summary: dict) -> None:
    """Print the dict returned by tiling.tile_class_balance()."""
    print(f"Tiles: {summary['n_tiles']}, "
          f"mean coverage: {summary['mean_coverage']:.1%}, "
          f"median coverage: {summary['median_coverage']:.1%}")
    for label, count in summary["bins"].items():
        pct = count / summary["n_tiles"] if summary["n_tiles"] else 0
        print(f"  {label:>7}: {count:5d} tiles ({pct:.1%})")


def plot_coverage_histogram(tile_gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Histogram of per-tile label coverage — reveals whether usable tiles
    are richly labeled or just barely scraping past the threshold."""
    fig, ax = plt.subplots()
    tile_gdf["label_coverage"].hist(bins=30, ax=ax)
    ax.set_xlabel("Fraction of tile covered by labels")
    ax.set_ylabel("Number of tiles")
    ax.set_title("Label coverage distribution")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def export_geojson(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    """Export a GeoDataFrame for visual inspection in QGIS. Skips silently
    if empty (e.g. no suspect geometries found) rather than erroring."""
    if len(gdf) == 0:
        print(f"Nothing to export to {out_path} (0 rows).")
        return
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved {out_path} ({len(gdf)} rows)")
