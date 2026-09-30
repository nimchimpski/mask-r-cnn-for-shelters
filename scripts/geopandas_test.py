import warnings
import matplotlib.pyplot as plot
import geopandas as gpd
import pandas as pd

import scripts.geo_mod as geo_mod


def main():
    shpf = "Struture_Footprint_March2020_C8E_C8W_C9/Structure_Footprint_March2020_C8E_C8W_C9.shp"
    camp_outlines = "kutupalong/Camp_09/CXB_New_camp_outline_KML_20180528.kml"

    gdf = gpd.read_file(shpf)

    # surveyed area = official outlines of the camps the labels cover (checked in QGIS)
    camps_gdf = gpd.read_file(camp_outlines).to_crs(gdf.crs)
    camps_gdf = camps_gdf[camps_gdf["Name"].isin(["Camp 8E", "Camp 8W", "Camp 9"])]
    aoi_poly = camps_gdf.geometry.union_all()

    camps_gdf.plot()

    # boundary_gdf.plot()

    #  create full tile grid
    tile_grid_gdf = geo_mod.generate_tile_grid(camps_gdf, 256, 0.3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in (within|intersection|unary_union)")

        # only tiles fully inside AOI
        filtered_tiles_gdf = tile_grid_gdf[tile_grid_gdf.within(aoi_poly)].copy()
        print(f"{len(filtered_tiles_gdf)} / {len(tile_grid_gdf)} tiles fall within the AOI")
        # union first: labels overlap, so summing per-label areas double-counts
        label_union = gdf.geometry.union_all()
        filtered_tiles_gdf["label_coverage"] = filtered_tiles_gdf.geometry.intersection(label_union).area / filtered_tiles_gdf.area

    min_coverage = 0.0
    useable_gdf = filtered_tiles_gdf[filtered_tiles_gdf["label_coverage"] > min_coverage]
    print(f"{len(useable_gdf)} usable tiles out of {len(filtered_tiles_gdf)} (threshold: {min_coverage:.0%} coverage)")

    # class balance: how coverage is spread across tiles
    cov_s = useable_gdf["label_coverage"]
    print(f"mean coverage: {cov_s.mean():.1%}, median coverage: {cov_s.median():.1%}")
    print(pd.cut(cov_s, [0, 0.05, 0.25, 0.5, 1], include_lowest=True).value_counts(sort=False))

    ax = camps_gdf.plot(color="lightgrey")
    # tile_grid_gdf.plot(ax=ax, facecolor="none", edgecolor="blue", linewidth=0.5)
    useable_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1)
    plot.show()




if __name__ == "__main__":
    main()