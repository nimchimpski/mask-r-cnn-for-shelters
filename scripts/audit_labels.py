"""Audit a label file against the drone images it should match.

Prints label health (CRS, polygon count, invalid/empty/multipart, value counts per field), then for
each camp tif: CRS, resolution, size, bands, nodata, date tags, number of intersecting polygons,
their area and side-in-pixels percentiles, and how much of the image the label hull covers.
Edit SHP and CAMPS below; run from the repo root.
"""
from pathlib import Path
import geopandas as gpd, rasterio, numpy as np
from shapely.geometry import box

ROOT = Path("data/kutupalong")
# SHP  = next((ROOT / "Struture_Footprint_March2020_C8E_C8W_C9").glob("*.shp"))
# SHP  = next((ROOT / "BGD_Camp_ShelterFootprint_UNOSAT_REACH_v1_January2019").glob("*.shp"))
SHP  = Path("labels_clean/cleaned.geojson")
CAMPS = ["Camp_08E", "Camp_08W", "Camp_09"]          # match your actual folder names/case

lab = gpd.read_file(SHP)
print("=== LABELS ===")
print("file:", SHP.name, "| polygons:", len(lab), "| CRS:", lab.crs)
print("invalid:", (~lab.is_valid).sum(), "| empty:", lab.is_empty.sum(),
      "| multipart:", (lab.geom_type == "MultiPolygon").sum())
for c in lab.columns.drop("geometry"):
    print(f"  field '{c}':", lab[c].value_counts(dropna=False).head(10).to_dict())

print("\n=== IMAGES ===")
for camp in CAMPS:
    for tif in sorted((ROOT / camp).rglob("*.tif")):
        with rasterio.open(tif) as src:
            l = lab.to_crs(src.crs)
            img = box(*src.bounds)
            inside = l[l.intersects(img)]
            area = inside.area
            side_px = np.sqrt(area) / src.res[0]
            tags = src.tags()
            print(f"\n[{camp}] {tif.name}")
            print("  CRS:", src.crs, "| res:", tuple(round(r, 3) for r in src.res),
                  "| size:", src.width, "x", src.height,
                  "| bands:", src.count, src.dtypes[0], "| nodata:", src.nodata)
            print("  date tags:", {k: v for k, v in tags.items() if "DATE" in k.upper()} or "none")
            print("  polygons intersecting:", len(inside))
            if len(inside):
                print("  area m² p1/25/50/75/99:", np.percentile(area, [1, 25, 50, 75, 99]).round(1))
                print("  side px p1/25/50/75/99:", np.percentile(side_px, [1, 25, 50, 75, 99]).round(1))
                cov = inside.unary_union.convex_hull.intersection(img).area / img.area
                print(f"  label hull covers {100*cov:.1f}% of image")