import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# === 1. File path configuration ===
csv_path = "./03_inference/cluster/topk_clusters.csv"
geojson_path = "./05_auxiliary/amazon_boundary.geojson"

# === 2. Load point CSV as GeoDataFrame ===
df = pd.read_csv(csv_path)
gdf_points = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"  # Assuming WGS84
)

# === 3. Load biome boundary GeoJSON ===
gdf_outline = gpd.read_file(geojson_path)

# === 4. Plot ===
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary
gdf_outline.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label='Amazon Biome Boundary')

# Plot points
gdf_points.plot(ax=ax, color='red', markersize=10, alpha=0.7, label='Cluster Centers')

# Beautify
ax.set_aspect('equal', adjustable='datalim')
ax.set_title("Top-K Cluster Centers within Amazon Biome", fontsize=15)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.tight_layout()
plt.show()
