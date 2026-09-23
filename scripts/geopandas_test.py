from geo_qa import validation, metrics, tiling, reporting


def main():
    shpf = "Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp"

    gdf = validation.load_shapefile(shpf)

    # print(gdf.head())

    gdf_addarea = metrics.compute_areas(gdf)
    # print(gdf_addarea.head(10))

    # clean, suspect = validation.repair_and_clean(gdf)
    # reporting.export_geojson(suspect, "xxx_suspects.geojson")
    # print(clean.head())

    metrics.dataset_extent_summary(gdf_addarea)

    tile_gdf = tiling.generate_tile_grid(gdf_addarea, 256, 0.3)
    tile_gdf = tiling.compute_tile_label_coverage(tile_gdf, gdf_addarea)
    usable = tiling.filter_usable_tiles(tile_gdf, 0.00)


if __name__ == "__main__":
    main()