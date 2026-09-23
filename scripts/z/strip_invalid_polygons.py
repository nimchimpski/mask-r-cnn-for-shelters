import geopandas as gpd
import numpy as np

gdf = gpd.read_file("Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp")
print(f"Total rows: {len(gdf)}")

# 1. Nulls and empties
null_geom = gdf.geometry.isna()
empty_geom = gdf.geometry.is_empty.fillna(False)
print(f"Null geometries: {null_geom.sum()}")
print(f"Empty geometries: {empty_geom.sum()}")

# 2. NaN coordinates (bounds will show NaN if coords are corrupted)
bounds = gdf.geometry.bounds
nan_bounds = bounds.isna().any(axis=1)
print(f"Geometries with NaN bounds: {nan_bounds.sum()}")

# 3. Geometry type check — mixed types can break union/intersection
print(f"Geometry types:\n{gdf.geometry.geom_type.value_counts(dropna=False)}")

# 4. is_valid, but safely (won't crash on the None/NaN cases above)
valid_check = gdf.geometry.apply(lambda g: g.is_valid if g is not None and not g.is_empty else False)
print(f"Invalid (non-null, non-empty) geometries: {(~valid_check[~null_geom & ~empty_geom]).sum()}")

# Combine every suspect flag
suspects_mask = null_geom | empty_geom | nan_bounds | (~valid_check)
suspects = gdf[suspects_mask]
clean = gdf[~suspects_mask].reset_index(drop=True)

print(f"\nSuspect geometries isolated: {len(suspects)}")
print(f"Clean geometries remaining: {len(clean)}")

# Save suspects for visual inspection in QGIS (only if there are any non-null ones to save)
non_null_suspects = suspects[~suspects.geometry.isna()]
if len(non_null_suspects) > 0:
    non_null_suspects.to_file("suspect_geometries.geojson", driver="GeoJSON")
    print("Saved suspect_geometries.geojson — open in QGIS to inspect")
else:
    print("All suspects have null geometry — nothing to visualize, just drop them")

# Save the cleaned version for your own downstream use
clean.to_file("Struture_Footprint_cleaned.geojson", driver="GeoJSON")
print("Saved Struture_Footprint_cleaned.geojson")