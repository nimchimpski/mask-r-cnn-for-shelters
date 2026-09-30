"""
geo_qa — the few pieces with enough logic to be worth reusing.
Everything else is inlined in the scripts (see analyze_camp.py).
"""

import geopandas as gpd
import pandas as pd
import rasterio
import torch
import numpy as np  

from shapely.geometry import box
from torch.utils.data import Dataset

DATA_DIR = "dataset_camps"

def diagnose_geometries(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, dict]:
    """
    Find geometries that would silently corrupt area sums, unions and
    intersections. `is_valid` alone misses null, empty and NaN-coordinate
    geometries, which make shapely quietly drop or corrupt results.

    Returns (suspects_mask, counts): True = suspect, plus a dict of tallies.

    Checks, in order:
      1. Null geometry (missing entirely)
      2. Empty geometry (zero-area / zero-vertex placeholder)
      3. NaN in the geometry's bounding box (corrupted coordinates)
      4. Topologically invalid (self-intersecting etc), checked only on
         rows that passed 1-3, since is_valid() can itself error on those.
    """
    null_geom = gdf.geometry.isna()
    empty_geom = gdf.geometry.is_empty.fillna(False)

    bounds = gdf.geometry.bounds
    nan_bounds = bounds.isna().any(axis=1)

    safe_rows = ~null_geom & ~empty_geom & ~nan_bounds
    invalid_topology = pd.Series(False, index=gdf.index)
    invalid_topology.loc[safe_rows] = ~gdf.geometry.loc[safe_rows].is_valid

    suspects = null_geom | empty_geom | nan_bounds | invalid_topology

    counts = {
        "null_geometries": int(null_geom.sum()),
        "empty_geometries": int(empty_geom.sum()),
        "nan_bounding_box": int(nan_bounds.sum()),
        "topologically_invalid": int(invalid_topology.sum()),
        "total_suspect": int(suspects.sum()),
        "total_rows": len(gdf),
    }

    return suspects, counts


def generate_tile_grid(
    gdf: gpd.GeoDataFrame, tile_size_px: int, resolution_m: float
) -> gpd.GeoDataFrame:
    """
    Build a regular grid of square tiles covering the bounding box of `gdf`.

    tile_size_px: tile side length in pixels (e.g. 256, matching your model
                  input size).
    resolution_m: ground sample distance in metres/pixel (e.g. 0.3 for PNEO,
                  0.5 for SkySat/Pléiades). tile_size_px * resolution_m gives
                  the tile's real-world side length in metres.
    """
    tile_size_m = tile_size_px * resolution_m
    minx, miny, maxx, maxy = gdf.total_bounds

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tiles.append(box(x, y, x + tile_size_m, y + tile_size_m))
            y += tile_size_m
        x += tile_size_m

    tile_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
    print(f"Generated {len(tile_gdf)} candidate tiles at {tile_size_m:.1f} m per side "
          f"({tile_size_px}px @ {resolution_m}m/px)")
    return tile_gdf


class TileDataset(Dataset):
    """torch's training loop needs a Dataset subclass. Masks already carry per-structure
    ids from the source shapefile (see build_dataset.py), so instances don't need
    reconstructing from a merged blob — touching structures stay separate."""

    # def __init__(self, names_list, augment=False):
    #     self.names_list = names_list
    #     self.augment = augment

    def __init__(self, names_list, augment=False, aug="flip", min_area=4):
        self.names_list, self.augment, self.aug, self.min_area = names_list, augment, aug, min_area


    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, i):
        name = self.names_list[i]
        with rasterio.open(f"{DATA_DIR}/images/{name}") as s:
            img_arr = s.read()
        with rasterio.open(f"{DATA_DIR}/masks/{name}") as s:
            instance_arr = s.read(1)
        image = torch.from_numpy(img_arr).float() / 255.0

        if self.augment:
            if np.random.random() < 0.5:
                image = image.flip(-1); instance_arr = instance_arr[:, ::-1]
            if np.random.random() < 0.5:
                image = image.flip(-2); instance_arr = instance_arr[::-1, :]
            if self.aug == "full":
                k = np.random.randint(4)
                image = torch.rot90(image, k, dims=(-2, -1))
                instance_arr = np.rot90(instance_arr, k)
                c = np.random.uniform(0.85, 1.15)   # contrast
                b = np.random.uniform(-0.08, 0.08)  # brightness, same shift on all bands
                m = image.mean()
                image = ((image - m) * c + m + b).clamp(0, 1)
            instance_arr = np.ascontiguousarray(instance_arr)

        boxes, masks = [], []
        for instance_id in np.unique(instance_arr):
            if instance_id == 0:
                continue
            mask_k = instance_arr == instance_id
            ys, xs = np.where(mask_k)
            if len(xs) < self.min_area:
                continue
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(mask_k)

        masks_arr = np.stack(masks) if masks else np.zeros((0, *instance_arr.shape), dtype=bool)
        boxes_arr = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        target = {
            "boxes": torch.from_numpy(boxes_arr),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks_arr.astype(np.uint8)),
            "image_id": torch.tensor([i]),
        }
        return image, target