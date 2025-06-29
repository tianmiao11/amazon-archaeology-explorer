import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def plot_merged_data_with_outline_result():
    geojson_path = "05_auxiliary/amazon_boundary.geojson"
    csv_path = "03_inference/cluster/matched_clusters_filter_positive_2km.csv"
    output_path = "04_visuals/plot_result.png"

    gdf_outline = gpd.read_file(geojson_path).to_crs(epsg=3857)
    gdf_outline["geometry"] = gdf_outline.buffer(10000)
    amazon_union = gdf_outline.geometry.unary_union

    df = pd.read_csv(csv_path)
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    gdf_points["in_amazon"] = gdf_points.geometry.to_crs(epsg=3857).within(amazon_union)

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_outline.to_crs(epsg=4326).plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)
    gdf_points.plot(ax=ax, markersize=30, color="red", label="Potential Archaeological Sites")

    xmin, ymin, xmax, ymax = gdf_outline.to_crs(epsg=4326).total_bounds
    margin_x = (xmax - xmin) * 0.05
    margin_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)

    ax.set_title("Spatial Distribution of Result", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_merged_data_with_outline_result()
