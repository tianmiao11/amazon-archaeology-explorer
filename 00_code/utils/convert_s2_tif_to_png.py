import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# === Convert single Sentinel-2 TIF to PNG ===
def s2_tif_to_png(tif_path, png_folder):
    with rasterio.open(tif_path) as src:
        if src.count < 3:
            raise ValueError(f"{os.path.basename(tif_path)} has less than 3 bands.")

        # Read RGB bands (B4, B3, B2 assumed in order)
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)

        # Normalize to [0, 1]
        rgb = np.stack([r, g, b], axis=-1) / 10000.0

        # Contrast stretch (2% to 98%)
        p2, p98 = np.nanpercentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)

        # Save as PNG
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        png_path = os.path.join(png_folder, f"{base_name}.png")
        plt.imsave(png_path, rgb)
        print(f"✅ Saved PNG: {png_path}")

# === Batch convert a folder of Sentinel-2 TIFs ===
def batch_convert_s2(folder_tif, folder_png):
    os.makedirs(folder_png, exist_ok=True)
    for fname in os.listdir(folder_tif):
        if fname.lower().endswith('.tif'):
            tif_path = os.path.join(folder_tif, fname)
            try:
                s2_tif_to_png(tif_path, folder_png)
            except Exception as e:
                print(f"❌ Failed: {fname} | {e}")

# === Set input/output folders ===
tif_folder = "./01_samples/tif/s2"
png_folder = "./04_visuals/png/s2"

batch_convert_s2(tif_folder, png_folder)
