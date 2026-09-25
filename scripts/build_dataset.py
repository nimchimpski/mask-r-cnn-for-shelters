import glob
import os
import shutil
import warnings

import geopandas as gpd
import matplotlib.pyplot as plot
import numpy as np
import rasterio
from osgeo import gdal
from rasterio.features import rasterize
from rasterio.windows import from_bounds as window_from_bounds

import geo_qa

LABELS = "Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp"
CAMP_OUTLINES = "kutupalong/Camp_09/CXB_New_camp_outline_KML_20180528.kml"
SURVEYED_CAMPS = ["Camp 8E", "Camp 8W", "Camp 9"]  # checked in QGIS: fully surveyed inside these outlines
IMAGES = sorted(glob.glob("kutupalong/Camp_*/Image_*.tif"))
MOSAIC = "kutupalong/mosaic.vrt"
MOSAIC_WARPED = "kutupalong/mosaic_warped.vrt"
TARGET_RES_M = None  # None = native pixel size; e.g. 0.3 to resample every source onto one shared grid
OUT_DIR = "dataset_camps"
TILE_PX = 256
BLOCK_M = 230  # split in ~230 m blocks so neighbouring tiles don't leak across splits
SEED = 0


def main():
    # spurious in shapely 2.0.1: inputs and results are valid
    warnings.filterwarnings("ignore", message="invalid value encountered in")

    # 1. labels + surveyed boundary (official camp outlines)
    gdf = gpd.read_file(LABELS)
    camps_gdf = gpd.read_file(CAMP_OUTLINES).to_crs(gdf.crs)
    camps_gdf = camps_gdf[camps_gdf["Name"].isin(SURVEYED_CAMPS)]
    aoi_poly = camps_gdf.geometry.union_all()

    # 2. merge camp images into one virtual mosaic (they share a pixel grid and agree where they overlap)
    gdal.BuildVRT(MOSAIC, IMAGES, srcNodata=0, VRTNodata=0).FlushCache()
    mosaic_path = MOSAIC
    if TARGET_RES_M is not None:
        # warp the merged mosaic, not IMAGES: a VRT-format warp only reads its first input
        # targetAlignedPixels puts pixel edges at multiples of the target res, so every source shares one grid
        # use resampleAlg="average" when shrinking, "bilinear"/"cubic" when enlarging
        gdal.Warp(MOSAIC_WARPED, MOSAIC, format="VRT", xRes=TARGET_RES_M, yRes=TARGET_RES_M, targetAlignedPixels=True,
                  resampleAlg="average", srcNodata=0, dstNodata=0).FlushCache()
        mosaic_path = MOSAIC_WARPED
    src = rasterio.open(mosaic_path)

    # 3. tile grid at the mosaic's resolution, snapped onto its pixel grid
    res, x0, y0 = src.res[0], src.transform.c, src.transform.f
    tile_gdf = geo_qa.generate_tile_grid(camps_gdf, TILE_PX, res)
    minx, miny = tile_gdf.total_bounds[:2]
    tile_gdf["geometry"] = tile_gdf.translate(-((minx - x0) % res), -((miny - y0) % res))
    tile_gdf = tile_gdf[tile_gdf.within(aoi_poly)].reset_index(drop=True)
    print(f"{len(tile_gdf)} tiles fall within the AOI")

    # 4. per tile: cut the chip (skip if any nodata), rasterise the labels
    # start clean so chips from a previous grid can't linger
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/masks", exist_ok=True)
    records = []
    for i, tile in enumerate(tile_gdf.geometry):
        # rounding only absorbs float error; tiles already sit on whole pixels
        win = window_from_bounds(*tile.bounds, transform=src.transform).round_offsets().round_lengths()
        if not (src.read_masks(1, window=win) > 0).all():
            continue
        chip = src.read(window=win)
        transform = src.window_transform(win)

        labels_here = gdf.geometry.iloc[gdf.sindex.query(tile, predicate="intersects")]
        if len(labels_here):
            mask = rasterize(((g, 1) for g in labels_here), out_shape=(TILE_PX, TILE_PX), transform=transform, dtype="uint8")
        else:
            mask = np.zeros((TILE_PX, TILE_PX), dtype="uint8")

        name = f"tile_{i:05d}.tif"
        profile = dict(driver="GTiff", width=TILE_PX, height=TILE_PX, crs=src.crs, transform=transform, compress="deflate")
        with rasterio.open(f"{OUT_DIR}/images/{name}", "w", count=chip.shape[0], dtype=chip.dtype, **profile) as dst:
            dst.write(chip)
        with rasterio.open(f"{OUT_DIR}/masks/{name}", "w", count=1, dtype="uint8", **profile) as dst:
            dst.write(mask, 1)
        records.append({"name": name, "label_frac": mask.mean(), "geometry": tile})

    src.close()
    print(f"Wrote {len(records)} / {len(tile_gdf)} chips (rest had nodata)")

    # 5. spatial train/val/test split by blocks of tiles, not per tile
    index_gdf = gpd.GeoDataFrame(records, crs=gdf.crs)
    index_gdf["block"] = [f"{int(t.bounds[0] // BLOCK_M)}_{int(t.bounds[1] // BLOCK_M)}" for t in index_gdf.geometry]
    blocks = index_gdf["block"].unique()
    np.random.default_rng(SEED).shuffle(blocks)
    n_train, n_val = int(len(blocks) * 0.7), int(len(blocks) * 0.15)
    split_of = {b: "train" for b in blocks[:n_train]}
    split_of |= {b: "val" for b in blocks[n_train:n_train + n_val]}
    split_of |= {b: "test" for b in blocks[n_train + n_val:]}
    index_gdf["split"] = index_gdf["block"].map(split_of)
    index_gdf.to_file(f"{OUT_DIR}/index.geojson")
    print(index_gdf.groupby("split").agg(tiles=("name", "size"), blocks=("block", "nunique"), label_frac=("label_frac", "mean")))

    # 6. eyeball a few pairs: red mask outlines should sit on roofs
    fig, axes = plot.subplots(1, 4, figsize=(16, 4))
    for ax, name in zip(axes, index_gdf.sample(4, random_state=SEED)["name"]):
        with rasterio.open(f"{OUT_DIR}/images/{name}") as s:
            ax.imshow(s.read().transpose(1, 2, 0))
        with rasterio.open(f"{OUT_DIR}/masks/{name}") as s:
            ax.contour(s.read(1), levels=[0.5], colors="red", linewidths=0.7)
        ax.set_title(name)
        ax.axis("off")
    fig.savefig(f"{OUT_DIR}/sample_pairs.png", dpi=100, bbox_inches="tight")
    plot.show()


if __name__ == "__main__":
    main()
