import os
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
from tqdm import tqdm
from scipy.ndimage import sobel, label
from PIL import Image

# === Parameters ===
tile_size_px = 10             # Patch size
ndvi_thr = 0.7                # NDVI threshold
rgb_std_thr = 0.005           # RGB grayscale std threshold
edge_thr_tile = 0.2           # Sobel edge threshold
large_area_frac = 0.0001      # Max edge-connected area fraction

# === Compute NDVI from Sentinel-2 patch ===
def compute_ndvi(da):
    red = da.sel(band='red').astype('float32') / 10000.0
    nir = da.sel(band='nir').astype('float32') / 10000.0
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

# === Generate speckle mask for one image ===
def get_speckled_mask(rgb_da, ndvi_da):
    R = rgb_da.sel(band='red').values / 10000.0
    G = rgb_da.sel(band='green').values / 10000.0
    B = rgb_da.sel(band='blue').values / 10000.0
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    ndvi = ndvi_da.values
    ndvi_mask = ndvi <= ndvi_thr
    ndvi[ndvi_mask] = np.nan
    gray[np.isnan(ndvi)] = np.nan

    H, W = gray.shape
    mask = np.zeros_like(gray, dtype=np.uint8)
    tile_area = tile_size_px * tile_size_px
    large_area_thresh_px = large_area_frac * tile_area

    for r in range(0, H, tile_size_px):
        for c in range(0, W, tile_size_px):
            ndvi_tile = ndvi[r:r + tile_size_px, c:c + tile_size_px]
            if np.isnan(ndvi_tile).all():
                continue

            gray_tile = gray[r:r + tile_size_px, c:c + tile_size_px]
            gray_filled = gray_tile.copy()
            gray_filled[np.isnan(gray_filled)] = 0.0

            gx = sobel(gray_filled, axis=1)
            gy = sobel(gray_filled, axis=0)
            grad_mag_tile = np.hypot(gx, gy)
            binary_edges_tile = (grad_mag_tile > edge_thr_tile).astype(np.uint8)

            labels_tile, num_labels_tile = label(binary_edges_tile)
            if num_labels_tile > 0:
                counts = np.bincount(labels_tile.ravel())
                max_comp_area = counts[1:].max() if len(counts) > 1 else 0
                if max_comp_area > large_area_thresh_px:
                    continue

            valid_gray_vals = gray_tile[~np.isnan(gray_tile)]
            if valid_gray_vals.size == 0:
                continue
            gray_std = valid_gray_vals.std()
            if gray_std <= rgb_std_thr:
                continue

            mask[r:r + tile_size_px, c:c + tile_size_px] = 1
    return mask

# === Batch process folder ===
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
    for f in tqdm(files, desc=f"Processing {os.path.basename(output_dir)}"):
        tif_path = os.path.join(input_dir, f)
        da = rxr.open_rasterio(tif_path)
        da = da.assign_coords(band=["blue", "green", "red", "nir"]).sel(band=["red", "green", "blue", "nir"])
        da = da.rename({"x": "x", "y": "y", "band": "band"})
        ndvi_da = compute_ndvi(da)
        mask = get_speckled_mask(da, ndvi_da)

        # Save as .npy
        npy_path = os.path.join(output_dir, f.replace(".tif", ".npy"))
        np.save(npy_path, mask)

        # Save as .png
        png_path = os.path.join(output_dir, f.replace(".tif", ".png"))
        img = Image.fromarray((mask * 255).astype(np.uint8))
        img.save(png_path)

# === Define paths ===
positive_dir = "./01_samples/tif/s2/positive"
negative_dir = "./01_samples/tif/s2/negative"
positive_out = "./01_samples/tif/s2_masked/positive"
negative_out = "./01_samples/tif/s2_masked/negative"

# === Run batch processing ===
process_folder(positive_dir, positive_out)
process_folder(negative_dir, negative_out)
