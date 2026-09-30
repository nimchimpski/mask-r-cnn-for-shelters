"""Run a saved Mask R-CNN checkpoint over dataset tiles and save detections as a
georeferenced GeoJSON (viewable in QGIS alongside labels_clean/cleaned.geojson).

Unlike eval_maskrcnn.py, this does not need ground truth, so it works on any split
(including "all") and reports no accuracy metrics.
"""
import argparse
import json
import sys

import geopandas as gpd
import rasterio
from pycocotools import mask as mask_utils
from rasterio.features import shapes
from shapely.geometry import shape

sys.path.insert(0, "scripts")
from eval_maskrcnn import load_model, run_inference, OBJ_SCORE_THRESH
from train_maskrcnn import DATA_DIR


def main(arg):
    # 1. pick tiles
    features_list = json.load(open(f"{DATA_DIR}/index.geojson"))["features"]
    if arg.split == "all":
        names_list = [f["properties"]["name"] for f in features_list]
    else:
        names_list = [f["properties"]["name"] for f in features_list if f["properties"]["split"] == arg.split]
    print(f"running inference on {len(names_list)} {arg.split} tiles")

    # 2. run the model, keep detections at or above the score threshold
    model = load_model(arg.ckpt)
    detections = [d for d in run_inference(model, names_list) if d["score"] >= arg.score_thresh]
    print(f"{len(detections)} detections >= score {arg.score_thresh}")

    # 3. group by tile so each tile's georeferencing is only read once
    dets_by_image = {}
    for det in detections:
        dets_by_image.setdefault(det["image_id"], []).append(det)

    # 4. turn each detection's pixel mask into a world-coordinate polygon
    records = []
    for image_id, dets_here in dets_by_image.items():
        name = names_list[image_id]
        with rasterio.open(f"{DATA_DIR}/images/{name}") as src:
            transform, crs = src.transform, src.crs
        for det in dets_here:
            binary_mask = mask_utils.decode(det["segmentation"])
            polys = [shape(geom) for geom, val in shapes(binary_mask, mask=binary_mask.astype(bool), transform=transform) if val == 1]
            poly = max(polys, key=lambda p: p.area)  # mask should be one blob; guards against stray speckle
            records.append({"geometry": poly, "tile": name, "score": det["score"]})

    # 5. save
    out_gdf = gpd.GeoDataFrame(records, crs=crs)
    out_path = f"tile_comparisons/{arg.run}_{arg.split}_predictions.geojson"
    out_gdf.to_file(out_path, driver="GeoJSON")
    print(f"saved {len(out_gdf)} polygons -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="label used for the output filename")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--score-thresh", type=float, default=OBJ_SCORE_THRESH)

    main(p.parse_args())
