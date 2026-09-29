import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import rgb_to_hsv

D = "dataset_camps"
RES = 0.1                                  # m/px; thresholds below assume native tiles
MIN_PX, MAX_PX = int(2 / RES**2), int(200 / RES**2)   # 2–200 m² in pixels
V_BRIGHT = 0.75                            # "roof-like" brightness; tune after viewing

idx = gpd.read_file(f"{D}/index.geojson")

# --- check A: label density vs neighbouring tiles
cent = idx.geometry.centroid
tile_w = idx.geometry.iloc[0].bounds[2] - idx.geometry.iloc[0].bounds[0]
nbr_med = []
for i, c in enumerate(cent):
    near = idx[(cent.distance(c) < 1.5 * tile_w) & (idx.index != i)]
    nbr_med.append(near["label_frac"].median() if len(near) else np.nan)
idx["nbr_med"] = nbr_med
idx["density_flag"] = (idx["nbr_med"] > 0.10) & (idx["label_frac"] < 0.3 * idx["nbr_med"])

# --- check B: bright shelter-sized blobs not touching any label
def unlabelled_blobs(name):
    with rasterio.open(f"{D}/images/{name}") as s:
        rgb = s.read().transpose(1, 2, 0) / 255.0
    with rasterio.open(f"{D}/masks/{name}") as s:
        lab = s.read(1) > 0
    bright = rgb_to_hsv(rgb)[..., 2] > V_BRIGHT
    near_label = ndimage.binary_dilation(lab, iterations=5)      # ~0.5 m margin
    cand = bright & ~near_label
    cand = ndimage.binary_opening(cand, iterations=2)            # drop speckle
    comp, n = ndimage.label(cand)
    sizes = ndimage.sum(cand, comp, range(1, n + 1))
    keep = [k + 1 for k, sz in enumerate(sizes) if MIN_PX <= sz <= MAX_PX]
    return len(keep), np.isin(comp, keep)

idx["n_blobs"] = [unlabelled_blobs(n)[0] for n in idx["name"]]

# --- rank and save
idx["score"] = idx["density_flag"].astype(int) * 10 + idx["n_blobs"]
ranked = idx.sort_values("score", ascending=False)
ranked.drop(columns="geometry").to_csv(f"{D}/qa_ranked.csv", index=False)
ranked.to_file(f"{D}/qa_candidates.geojson")                     # open in QGIS
print(ranked[["name", "split", "label_frac", "nbr_med", "density_flag", "n_blobs"]].head(20))
print("\nflagged per split:\n", ranked[ranked["score"] > 0].groupby("split").size())

# --- contact sheet of the top 24: red = labels, cyan = unlabelled bright blobs
fig, axes = plt.subplots(4, 6, figsize=(24, 16))
for ax, (_, r) in zip(axes.ravel(), ranked.head(24).iterrows()):
    with rasterio.open(f"{D}/images/{r['name']}") as s:
        ax.imshow(s.read().transpose(1, 2, 0))
    with rasterio.open(f"{D}/masks/{r['name']}") as s:
        ax.contour(s.read(1) > 0, levels=[0.5], colors="red", linewidths=0.6)
    ax.contour(unlabelled_blobs(r["name"])[1], levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_title(f"{r['name']} [{r['split']}] blobs={r['n_blobs']}"
                 f"{' DENSITY' if r['density_flag'] else ''}", fontsize=8)
    ax.axis("off")
fig.savefig(f"{D}/qa_top24.png", dpi=90, bbox_inches="tight")