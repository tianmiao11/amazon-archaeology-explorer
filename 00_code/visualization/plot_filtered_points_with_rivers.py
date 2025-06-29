import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from tqdm import tqdm

# -------- Path Configuration --------
csv_path = "./03_inference/cluster/matched_clusters_filter_positive_2km.csv"
river_path = "./05_auxiliary/amazon_river.gpkg"
outline_path = "./05_auxiliary/amazon_boundary.geojson"

# -------- Load biome boundary --------
print("🌍 Loading Amazon biome boundary...")
gdf_outline = gpd.read_file(outline_path)
gdf_outline = gdf_outline.to_crs(epsg=4326)

# -------- Load river shapefile --------
print("🌊 Loading river network...")
gdf_river = gpd.read_file(river_path)
gdf_river = gdf_river.to_crs(epsg=4326)

# -------- Load cluster points --------
print("📍 Loading matched cluster points...")
df = pd.read_csv(csv_path)
print("📋 Columns:", list(df.columns))

# Convert to GeoDataFrame
tqdm.pandas(desc="🧱 Creating Point Geometry")
df["geometry"] = list(tqdm([Point(xy) for xy in zip(df["lon"], df["lat"])], desc="🧱 Creating Points", unit="pt"))
gdf_points = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# -------- Plotting --------
print("🖼️ Plotting figure...")
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary
gdf_outline.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, label='Amazon Boundary')

# Plot river lines
gdf_river.plot(ax=ax, color='blue', linewidth=0.5, label='Rivers')

# Plot matched filtered points
gdf_points.plot(ax=ax, color='red', markersize=6, label='Matched Cluster Points')

# Labels and legend
plt.title("Matched Clusters (DEM ∩ S2), >2km from River", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
