"""
geo_qa.reporting
----------------
Diagnostic outputs meant for a human to actually look at — a coverage
histogram (is coverage uniform or lopsided?) and exportable GeoJSON layers
(suspect geometries, cleaned data) to drag straight into QGIS.
"""

import geopandas as gpd
import matplotlib.pyplot as plt


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
