import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.train_maskrcnn import TileDataset   # your filename
feats = json.load(open("dataset_camps/index.geojson"))["features"]
names = [f["properties"]["name"] for f in feats if f["properties"]["split"] == "train"]
ds = TileDataset(names, augment=True, aug="full")
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for a in ax.ravel():
    img, t = ds[0]
    a.imshow(img.permute(1, 2, 0))
    a.contour(t["masks"].sum(0).numpy() > 0, levels=[0.5], colors="r", linewidths=0.6)
    a.axis("off")
fig.savefig("aug_check.png", dpi=100)