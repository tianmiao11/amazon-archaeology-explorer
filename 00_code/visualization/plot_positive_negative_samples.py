import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# === Plot positive samples grouped by Source ID ===
def plot_merged_data_with_outline():
    geojson_path = "05_auxiliary/amazon_boundary.geojson"
    excel_path = "01_samples/data_merged_final.xlsx"
    output_path = "04_visuals/plot_positive_samples.png"

    gdf_outline = gpd.read_file(geojson_path)
    gdf_outline["geometry"] = gdf_outline.buffer(0.1)
    amazon_union = gdf_outline.unary_union

    df = pd.read_excel(excel_path)
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"
    )
    gdf_points["in_amazon"] = gdf_points.geometry.within(amazon_union)

    source_ids = sorted(gdf_points["Source ID"].unique())
    colors = plt.cm.get_cmap("tab10", len(source_ids))

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_outline.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5)

    for i, sid in enumerate(source_ids):
        gdf_points[gdf_points["Source ID"] == sid].plot(
            ax=ax,
            markersize=30,
            label=f"Source ID {sid}",
            color=colors(i)
        )

    xmin, ymin, xmax, ymax = gdf_outline.total_bounds
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)

    ax.set_title("Spatial Distribution of Positive Samples", fontsize=16)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Source ID", loc="upper right")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# === Plot all samples grouped by Type + Description ===
def plot_by_type_description_with_outline():
    geojson_path = "05_auxiliary/amazon_boundary.geojson"
    excel_path = "01_samples/data_merged.xlsx"
    output_path = "04_visuals/plot_samples_classify.png"

    gdf_outline = gpd.read_file(geojson_path)
    df = pd.read_excel(excel_path)
    df["Description"] = df["Description"].fillna("none")

    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326"
    )

    gdf_points["Group"] = gdf_points["Type"] + " | " + gdf_points["Description"]
    group_keys = sorted(gdf_points["Group"].unique())
    types = sorted(gdf_points["Type"].unique())

    type_color_map = {
        t: to_rgb(c) for t, c in zip(types, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])
    }

    group_color_map = {}
    for t in types:
        descs = sorted(gdf_points[gdf_points["Type"] == t]["Description"].unique())
        n = len(descs)
        for i, d in enumerate(descs):
            base_rgb = type_color_map[t]
            factor = 0.5 + 0.5 * (i / max(n - 1, 1))
            adjusted_rgb = tuple(min(1.0, c * factor) for c in base_rgb)
            group_color_map[f"{t} | {d}"] = adjusted_rgb

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf_outline.plot(ax=ax, color="none", edgecolor="black", linewidth=0.5)

    for group in group_keys:
        color = group_color_map[group]
        gdf_points[gdf_points["Group"] == group].plot(
            ax=ax,
            markersize=30,
            label=group,
            color=color
        )

    xmin, ymin, xmax, ymax = gdf_outline.total_bounds
    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 1, ymax + 1)

    ax.set_title("Amazon Sites Grouped by Type + Description", fontsize=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Type | Description", loc="upper right", fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_merged_data_with_outline()
    # plot_by_type_description_with_outline()
