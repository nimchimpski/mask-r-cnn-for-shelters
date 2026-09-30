import sys, json, numpy as np
# sys.path.insert(0, "scripts")
from scripts.train_maskrcnn import TileDataset
feats = json.load(open("dataset_camps/index.geojson"))["features"]
names = [f["properties"]["name"] for f in feats if f["properties"]["split"] == "val"]
ds = TileDataset(names)
areas = np.concatenate([ds[i][1]["masks"].sum((1, 2)).numpy() for i in range(len(ds))])
print(len(areas), int((areas < 25).sum()), f"{(areas < 25).mean():.1%}")