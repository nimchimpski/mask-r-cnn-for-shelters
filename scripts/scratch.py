import geopandas as gpd
idx = gpd.read_file("dataset_camps/qa_candidates.geojson")
print(idx.groupby("split")["density_flag"].sum())


