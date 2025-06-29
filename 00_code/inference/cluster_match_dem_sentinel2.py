import pandas as pd
from geopy.distance import geodesic
from tqdm import tqdm

# === 1. File paths ===
path_dem = "./03_inference/cluster/cluster_nasadem.csv"
path_s2  = "./03_inference/cluster/cluster_sentinel2.csv"
output_path = "./03_inference/cluster/matched_clusters.csv"

# === 2. Read input CSV files ===
df_dem = pd.read_csv(path_dem, sep=",")
df_s2  = pd.read_csv(path_s2, sep=",")

# === 3. Match DEM and S2 points (distance < 1 km) ===
matched = []

for i, dem_row in tqdm(df_dem.iterrows(), total=len(df_dem), desc="Matching DEM points"):
    dem_point = (dem_row["lat"], dem_row["lon"])
    for _, s2_row in df_s2.iterrows():
        s2_point = (s2_row["lat"], s2_row["lon"])
        distance = geodesic(dem_point, s2_point).km
        if distance < 1.0:
            print(f"✅ Match found: DEM No {dem_row['NO']} <--> S2 No {s2_row['NO']}")
            matched.append({
                "NO": dem_row["NO"],
                "lon": dem_row["lon"],
                "lat": dem_row["lat"],
                "prob_dem": dem_row["prob_mean"],
                "prob_s2": s2_row["prob"]
            })
            break  # Only one match per DEM point

# === 4. Save matched results ===
df_matched = pd.DataFrame(matched)
df_matched.to_csv(output_path, index=False)
print(f"\n✅ Done! Matched {len(df_matched)} points. Saved to: {output_path}")
