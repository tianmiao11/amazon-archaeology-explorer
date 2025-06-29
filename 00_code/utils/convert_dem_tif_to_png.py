import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# === Convert single DEM TIF to PNG ===
def dem_tif_to_png(tif_path, png_folder):
    with rasterio.open(tif_path) as src:
        # 读取单波段
        data = src.read(1).astype(np.float32)

        # 拉伸对比度（2%~98%）
        p2, p98 = np.nanpercentile(data, (2, 98))
        data_stretched = np.clip((data - p2) / (p98 - p2), 0, 1)

        # 保存为 PNG，使用地形色表
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        png_path = os.path.join(png_folder, f"{base_name}.png")
        plt.imsave(png_path, data_stretched, cmap="terrain")
        print(f"✅ Saved PNG: {png_path}")

# === Batch convert a folder of DEM TIFs ===
def batch_convert_dem(folder_tif, folder_png):
    os.makedirs(folder_png, exist_ok=True)
    for fname in os.listdir(folder_tif):
        if fname.lower().endswith('.tif'):
            tif_path = os.path.join(folder_tif, fname)
            try:
                dem_tif_to_png(tif_path, folder_png)
            except Exception as e:
                print(f"❌ Failed: {fname} | {e}")

# === Set input/output folders ===
tif_folder = "./01_samples/tif/dem"
png_folder = "./04_visuals/png/dem"

batch_convert_dem(tif_folder, png_folder)
