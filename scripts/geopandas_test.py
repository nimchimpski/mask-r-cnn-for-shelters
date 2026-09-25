import warnings
import shapely
import matplotlib.pyplot as plot
import geopandas as gpd

from geo_qa import validation, metrics, tiling, reporting


def main():
    shpf = "Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp"

    gdf = validation.load_shapefile(shpf)

    with warnings.catch_warnings():
        # spurious in shapely 2.0.1: inputs and result are valid
        warnings.filterwarnings("ignore", message="invalid value encountered in (buffer|unary_union)")
        # buffer each structure 40 m outward; overlapping buffers merge into one shape per cluster
        boundary = gdf.geometry.buffer(40).union_all()
        # can make >1 polygon
        # take the largest polygon
        largest_poly = max(shapely.get_parts(boundary), key=lambda g: g.area)
        # shrink back the buffer, then make it a gdf
        largest_gdf = gpd.GeoDataFrame(geometry=[largest_poly.buffer(-40)], crs=gdf.crs)

    largest_gdf.plot()
    # boundary_gdf.plot()

    tile_grid_gdf = tiling.generate_tile_grid(largest_gdf, 256, 0.3)
    filtered_tiles_gdf = tiling.filter_tiles_within_aoi(tile_grid_gdf, largest_gdf, mode="contains")
    filtered_tiles_gdf = tiling.compute_tile_label_coverage(filtered_tiles_gdf, gdf)

    useable_gdf = tiling.filter_usable_tiles(filtered_tiles_gdf, min_coverage=0.0)

    balance = tiling.tile_class_balance(useable_gdf)
    reporting.print_class_balance(balance)

    ax = largest_gdf.plot(color="lightgrey")
    # tile_grid_gdf.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=0.5)
    useable_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1)
    plot.show()

    return


    gdf_addarea = metrics.add_areas(gdf)


    summary = metrics.dataset_extent_summary(gdf_addarea)
    print(">>>\n",summary)


    # plot.show() 


    tile_gdf = tiling.generate_tile_grid(gdf_addarea, 256, 0.3)
    tile_gdf = tiling.add_tile_label_coverage(tile_gdf, gdf_addarea)
    usable = tiling.filter_usable_tiles(tile_gdf, 0.00)


if __name__ == "__main__":
    main()