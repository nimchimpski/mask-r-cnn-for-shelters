# /Users/alexwebb/laptop_coding/MSF call/Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp


import geopandas as gpd
import numpy as np
from shapely.geometry import box

gdf = gpd.read_file("Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp")

invalid_before = (~gdf.is_valid).sum()
gdf["geometry"] = gdf.geometry.buffer(0)

print(f"Invalid geometries found: {invalid_before}")

minx, miny, maxx, maxy = gdf.total_bounds
print(f"Bounding box: {(maxx-minx):.1f} m x {(maxy-miny):.1f} m "
      f"= {((maxx-minx)*(maxy-miny))/1e6:.3f} km²")

hull = gdf.union_all().convex_hull
print(f"Convex hull area: {hull.area/1e6:.3f} km²")

print(f"Total footprint (tent) area: {gdf.geometry.area.sum():.1f} m²")
print(f"Number of labeled structures: {len(gdf)}")

tile_size = 256 * 0.3  # e.g. 256px tiles at 30cm resolution, in metres
tiles = []
x = minx
while x < maxx:
    y = miny
    while y < maxy:
        tiles.append(box(x, y, x + tile_size, y + tile_size))
        y += tile_size
    x += tile_size

tile_gdf = gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)
# fraction of each tile actually covered by labeled polygons
tile_gdf["label_coverage"] = tile_gdf.geometry.apply(
    lambda t: gdf.geometry.intersection(t).area.sum() / t.area
)
usable = tile_gdf[tile_gdf["label_coverage"] > 0.05]  # threshold, adjust as needed
print(f"{len(usable)} usable tiles out of {len(tile_gdf)}")

# distribution of coverage across all tiles
import matplotlib.pyplot as plt
tile_gdf["label_coverage"].hist(bins=30)
plt.xlabel("Fraction of tile covered by labels")
plt.ylabel("Number of tiles")
plt.title("Label coverage distribution")
plt.savefig("coverage_histogram.png", dpi=120)
print("Saved coverage_histogram.png")