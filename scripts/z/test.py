import geopandas as gpd
import numpy as np

gdf = gpd.read_file("Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp")
print(f"Total rows: {len(gdf)}")

print('>>>', gdf.columns.tolist())      # what attribute fields exist
print('>>>', gdf.head())                # first few rows, geometry included
print('>>>', gdf.geometry.geom_type.unique())  # Polygon? MultiPolygon? mixed?
print('>>>', gdf.crs)                   # coordinate reference system
print('>>>', gdf.dtypes) 