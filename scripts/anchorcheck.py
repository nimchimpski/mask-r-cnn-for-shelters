"""Compare how well two anchor-size configs cover the dataset's structures.

Builds the default (32-512) and custom (16-256) Mask R-CNN anchors for a 512 px tile, takes the
instance bounding boxes (>= 25 px) from every 3rd mask in dataset_camps/masks, and prints each
config's best-anchor IoU per box: median, share below 0.5, and the same for small boxes (< 32 px side).
Run from the repo root.
"""
import json, glob, numpy as np, rasterio, torch
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

def anchors_for(sizes, px=512):
    gen = AnchorGenerator(sizes=tuple((s,) for s in sizes), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    feats = [torch.zeros(1, 256, px // s, px // s) for s in (4, 8, 16, 32, 64)]   # FPN strides
    return gen(ImageList(torch.zeros(1, 3, px, px), [(px, px)]), feats)[0]

configs = {"default 32-512": anchors_for((32, 64, 128, 256, 512)),
           "custom 16-256":  anchors_for((16, 32, 64, 128, 256))}

boxes = []
for f in sorted(glob.glob("dataset_camps/masks/*.tif"))[::3]:          # every 3rd tile is plenty
    m = rasterio.open(f).read(1)
    for k in np.unique(m)[1:]:
        ys, xs = np.where(m == k)
        if len(xs) >= 25:
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
b = torch.tensor(boxes, dtype=torch.float32)
side = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sqrt()

for name, a in configs.items():
    best = box_iou(b, a).max(1).values
    small = side < 32
    print(f"{name}: median best IoU {best.median():.2f} | best IoU < 0.5: {(best < 0.5).float().mean():.1%} "
          f"| small (<32px) median {best[small].median():.2f}, <0.5: {(best[small] < 0.5).float().mean():.1%}")
