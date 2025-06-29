import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# -------------------------------------
# ✅ 1. Parameter settings
# -------------------------------------
heatmap_path = "./04_visuals/prediction_heatmap_v7.tif"
prob_threshold = 0.9
eps_pixels = 32
min_samples = 3
top_k_clusters = 1000
tile_size = 2048  # Size of each non-overlapping tile
output_csv_path = "./03_inference/cluster/topk_clusters_partitioned.csv"

# -------------------------------------
# ✅ 2. Open raster heatmap
# -------------------------------------
with rasterio.open(heatmap_path) as src:
    transform = src.transform
    width, height = src.width, src.height

    cluster_centers = []

    # Traverse all tiles (non-overlapping)
    for row_start in tqdm(range(0, height, tile_size), desc="🔍 Scanning tile rows"):
        for col_start in range(0, width, tile_size):
            win_h = min(tile_size, height - row_start)
            win_w = min(tile_size, width - col_start)
            window = Window(col_start, row_start, win_w, win_h)
            block = src.read(1, window=window)

            # Extract high-confidence pixels within the tile
            rows, cols = np.where(block > prob_threshold)
            if len(rows) == 0:
                continue

            points_pixel = np.column_stack((cols + col_start, rows + row_start))
            probs = block[rows, cols]

            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=eps_pixels, min_samples=min_samples)
            labels = clustering.fit_predict(points_pixel)

            # Extract cluster centers within this tile
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue  # Ignore noise
                mask = labels == cluster_id
                cluster_pts = points_pixel[mask]
                cluster_probs = probs[mask]
                n_points = len(cluster_pts)
                prob_max = cluster_probs.max()
                prob_mean = cluster_probs.mean()
                max_idx = np.argmax(cluster_probs)
                col, row = cluster_pts[max_idx]
                lon, lat = rasterio.transform.xy(transform, row, col, offset='center')
                cluster_centers.append({
                    'lon': lon,
                    'lat': lat,
                    'prob_max': prob_max,
                    'prob_mean': prob_mean,
                    'n_points': n_points
                })

# -------------------------------------
# ✅ 3. Sort and save top-K cluster centers
# -------------------------------------
df_clusters = pd.DataFrame(cluster_centers)
df_clusters.sort_values(by="prob_max", ascending=False, inplace=True)
topk_df = df_clusters.head(top_k_clusters).reset_index(drop=True)

# Save as CSV
topk_df[["lon", "lat", "prob_max", "prob_mean", "n_points"]].to_csv(output_csv_path, index=False)
print(f"\n✅ Saved Top-{top_k_clusters} cluster centers to: {output_csv_path}")
