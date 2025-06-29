# 📁 `00_code/` - Project Scripts Overview

This directory contains all core scripts for training, inference, utility processing, and visualization related to archaeological site detection in the Amazon biome using DEM and Sentinel-2 imagery.

---

## 🧪 `train/` - Model Training

| Script | Description |
|--------|-------------|
| `train_dem_resnet.py` | Train a ResNet model using 4-channel DEM-derived features (elevation, slope, curvature, LRM). |
| `train_s2_resnet.py`  | Train a CNN model using Sentinel-2 RGB image patches. |

---

## 📤 `inference/` - Inference & Clustering

| Script | Description |
|--------|-------------|
| `infer_heatmap_from_dem.py`     | Perform sliding window inference on DEM tiles to generate a prediction heatmap. |
| `heatmap_partitioned.py`        | Split a large heatmap into subregions and apply DBSCAN clustering within each partition. |
| `cluster_match_dem_sentinel2.py`| Match prediction clusters from DEM and Sentinel-2 results that lie within 2 km of each other. |
| `filter_points_near_river.py`   | Filter out candidate points that are within 2 km of river features to reduce false positives. |
| `cluster.py`                    | Cluster high-probability prediction points using DBSCAN to extract site candidates. |

---

## 🛠️ `utils/` - Utility Tools

| Script | Description |
|--------|-------------|
| `convert_dem_tif_to_png.py` | Convert DEM GeoTIFFs to PNG images using the "terrain" colormap for visualization. |
| `s2_tif_to_png.py`          | Convert Sentinel-2 RGB TIFs to PNG images with contrast stretch. |
| `generate_speckle_mask.py`  | Generate speckled anomaly masks using NDVI and local texture analysis (gray variance + edge structure). |

---

## 🧱 `visualization/` - Visualization Scripts

| Script | Description |
|--------|-------------|
| `plot_clusters_with_outline.py`         | Plot clustering results with the Amazon biome boundary overlay. |
| `plot_filtered_points_with_rivers.py`   | Visualize filtered candidate points alongside river networks. |
| `plot_final_matched_results.py`         | Plot final matched results after combining DEM and Sentinel-2 clusters. |
| `plot_positive_negative_samples.py`     | Visualize the spatial distribution of positive and negative training samples. |
| `visualize_dem_s2_tree_3d.py`           | Build and render a 3D model with DEM terrain and NDVI-derived tree cones using `trimesh`. |

---

For project execution and dataset organization, please refer to the main `README.md` at the root of the repository.

