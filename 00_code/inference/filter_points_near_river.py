import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

tqdm.pandas()  # Enable tqdm progress bars for Pandas

# ✅ Step 1: Load river polyline data
print("📂 Loading river vector file...")
river_lines = gpd.read_file("./05_auxiliary/amazon_river.gpkg")
print(f"✅ Loaded {len(river_lines)} river lines")

# ✅ Step 2: Project to meter-based CRS (Web Mercator)
print("📐 Projecting to EPSG:3857...")
river_lines = river_lines.to_crs(epsg=3857)

# ✅ Step 3: Build 2km buffer zone
print("🛡️ Generating 2 km buffer zones...")
buffer_list = list(tqdm(river_lines.geometry, desc="🚧 Buffering", unit="line"))
buffered = [geom.buffer(2000) for geom in buffer_list]
buffer_union = gpd.GeoSeries(buffered).unary_union
print("✅ Buffer union completed")

# ✅ Step 4: Read candidate points and convert to GeoDataFrame
print("📍 Reading candidate points CSV...")
df = pd.read_csv("./03_inference/cluster/topk_clusters.csv", sep=',', encoding='utf-8-sig')
print(f"✅ Loaded {len(df)} points")

print("🧭 Creating point geometries...")
df["geometry"] = list(tqdm([Point(xy) for xy in zip(df["lon"], df["lat"])], desc="🧱 Creating Points", unit="pt"))
gdf_points = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_crs(epsg=3857)

# ✅ Step 5: Check whether points fall within buffer zone
print("🔍 Checking whether each point is within 2km buffer...")
gdf_points["in_buffer"] = gdf_points["geometry"].progress_apply(lambda geom: geom.within(buffer_union))

# ✅ Step 6: Filter out points too close to rivers
print("🚫 Removing points <2km from river...")
filtered = gdf_points[~gdf_points["in_buffer"]].copy()
print(f"✅ Remaining points after filtering: {len(filtered)}")

# ✅ Step 7: Add index (NO) and save result
print("💾 Saving filtered results...")
filtered = filtered.reset_index(drop=True)
filtered.insert(0, "NO", filtered.index + 1)
filtered.drop(columns=["geometry", "in_buffer"], inplace=True)
filtered.to_csv("./03_inference/cluster/topk_clusters_filtered.csv", index=False)
print("✅ Results saved to topk_clusters_filtered.csv")
