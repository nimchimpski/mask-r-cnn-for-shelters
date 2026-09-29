"""
clean_labels.py
---------------
One place where structure-footprint labels get cleaned. The tiling script reads its output.

Steps: load -> reproject -> clip to surveyed camps -> fix geometry -> split multipart ->
drop slivers / exact duplicates -> class filtering -> flag heavy overlaps -> write outputs.

Usage:
    python clean_labels.py path/to/footprints.shp \
        --camp-outlines data/kutupalong/Camp_09/CXB_New_camp_outline_KML_20180528.kml \
        --camps "Camp 8E" "Camp 8W" "Camp 9" --small class --out-dir labels_clean

Outputs (in --out-dir):
    cleaned.geojson   polygons to train on, with a `class_id` column (1 = structure, 2 = small structure)
    suspects.geojson  everything removed or flagged, with a `reason` column -- open in QGIS
Dataset statistics (tile counts, coverage, density) come from the tiling script's index.geojson,
not from here, because only the tiler knows which tiles survived the nodata check.
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid

TARGET_CRS = "EPSG:32646"          # UTM 46N, metres
MIN_AREA_M2 = 0.5                  # below this: slivers left by make_valid / clipping, not structures
SMALL_AREA_M2 = 5.0                # matches the Jan 2019 'area_class' rule (< 5 m2: likely latrine/tubewell/shower)
OVERLAP_IOU_FLAG = 0.5             # pairs overlapping more than this are flagged (not removed) for review


def flag(gdf, reason):
    out = gdf.copy()
    out["reason"] = reason
    return out


def main(shp, outlines, camps, small_mode, out_dir):
    warnings.filterwarnings("ignore", message="invalid value encountered in")  # spurious in shapely 2.0.x
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suspects = []

    # 1. load + reproject
    gdf = gpd.read_file(shp).to_crs(TARGET_CRS)
    print(f"loaded {len(gdf)} polygons from {Path(shp).name}")

    # 2. clip to surveyed camps (keep polygons that touch the AOI; the tiler only cuts tiles inside it)
    camps_gdf = gpd.read_file(outlines).to_crs(TARGET_CRS)
    missing = set(camps) - set(camps_gdf["Name"])
    if missing:
        raise ValueError(f"camp names not in outline file: {missing}")
    aoi = camps_gdf[camps_gdf["Name"].isin(camps)].geometry.union_all()
    gdf = gdf[gdf.intersects(aoi)].copy()
    print(f"{len(gdf)} polygons touch {', '.join(camps)}")

    # 3. fix invalid geometry
    invalid = ~gdf.is_valid
    if invalid.any():
        suspects.append(flag(gdf[invalid], "was_invalid_fixed"))
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)

    # 4. split multipart / collections into single polygons; keep polygons only
    n_multi = (gdf.geom_type != "Polygon").sum()
    gdf = gdf.explode(index_parts=False)
    gdf = gdf[gdf.geom_type == "Polygon"].reset_index(drop=True)
    gdf = gdf[~gdf.is_empty]

    # 5. drop slivers (areas recomputed from geometry; stored area fields are ignored)
    gdf["area_m2"] = gdf.geometry.area
    tiny = gdf["area_m2"] < MIN_AREA_M2
    suspects.append(flag(gdf[tiny], "sliver_lt_0.5m2"))
    gdf = gdf[~tiny]

    # 6. drop exact duplicate geometries (normalised so vertex order / start point don't matter)
    dup = gdf.geometry.normalize().to_wkb().duplicated()
    suspects.append(flag(gdf[dup.values], "exact_duplicate"))
    gdf = gdf[~dup.values]

    # 7. class filtering (only where the fields exist -- the March 2020 file has none of these)
    if "un_class" in gdf.columns:
        bridge = gdf["un_class"].eq("Bridge")
        suspects.append(flag(gdf[bridge], "bridge"))
        gdf = gdf[~bridge]
        n_nan = gdf["un_class"].isna().sum()
        if n_nan:
            print(f"note: {n_nan} polygons with no un_class kept (check img_src/dig_by in suspects if unsure)")
            suspects.append(flag(gdf[gdf["un_class"].isna()], "no_un_class_kept"))

    is_small = gdf["area_m2"] < SMALL_AREA_M2
    if small_mode == "drop":
        # warning: visible roofs become background in training
        suspects.append(flag(gdf[is_small], "small_dropped"))
        gdf = gdf[~is_small]
        gdf["class_id"] = 1
    elif small_mode == "class":
        gdf["class_id"] = is_small.map({True: 2, False: 1}).astype(int)
    else:  # keep as one class
        gdf["class_id"] = 1
    gdf = gdf.reset_index(drop=True)

    # 8. flag (don't remove) heavily overlapping pairs: possible near-duplicates or double digitising
    pairs = gpd.sjoin(gdf[["geometry"]], gdf[["geometry"]], predicate="intersects")
    pairs = pairs[pairs.index < pairs["index_right"]]
    if len(pairs):
        a = gdf.geometry.loc[pairs.index].values
        b = gdf.geometry.loc[pairs["index_right"]].values
        inter = gpd.GeoSeries(a).intersection(gpd.GeoSeries(b)).area.values
        union = gpd.GeoSeries(a).union(gpd.GeoSeries(b)).area.values
        heavy = pairs[(inter / union) > OVERLAP_IOU_FLAG]
        ids = sorted(set(heavy.index) | set(heavy["index_right"]))
        if ids:
            suspects.append(flag(gdf.loc[ids], "overlap_iou_gt_0.5_kept"))

    # 9. write + summary
    gdf.to_file(out_dir / "cleaned.geojson", driver="GeoJSON")
    sus = pd.concat([s for s in suspects if len(s)], ignore_index=True) if any(len(s) for s in suspects) else None
    if sus is not None:
        gpd.GeoDataFrame(sus, crs=TARGET_CRS).to_file(out_dir / "suspects.geojson", driver="GeoJSON")

    print(f"\nmultipart/collections split: {n_multi}")
    if sus is not None:
        print(sus["reason"].value_counts().to_string())
    print(f"\nkept {len(gdf)} polygons -> {out_dir / 'cleaned.geojson'}")
    print(gdf["class_id"].value_counts().rename({1: "structure", 2: "small (<5 m2)"}).to_string())
    print("area m2 p1/25/50/75/99:",
          gdf["area_m2"].quantile([.01, .25, .5, .75, .99]).round(1).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("shapefile")
    p.add_argument("--camp-outlines", default="data/kutupalong/Camp_09/CXB_New_camp_outline_KML_20180528.kml")
    p.add_argument("--camps", nargs="+", default=["Camp 8E", "Camp 8W", "Camp 9"])
    p.add_argument("--small", choices=["keep", "class", "drop"], default="keep",
                   help="<5 m2 structures: keep in one class (default), make class 2, or drop")
    p.add_argument("--out-dir", default="labels_clean")
    a = p.parse_args()
    main(a.shapefile, a.camp_outlines, a.camps, a.small, a.out_dir)
