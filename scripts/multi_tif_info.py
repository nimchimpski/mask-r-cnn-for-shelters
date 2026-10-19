import rasterio, pathlib, csv

rows = []
for p in pathlib.Path("data/kutupalong").rglob("*.tif"):
    with rasterio.open(p) as ds:
        rows.append({
            "file": str(p),
            "epsg": ds.crs.to_epsg() if ds.crs else None,
            "res_x": ds.res[0],
            "res_y": ds.res[1],
            "origin_x": ds.transform.c,
            "origin_y": ds.transform.f,
            "width": ds.width,
            "height": ds.height,
        })

with open("tif_info.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)