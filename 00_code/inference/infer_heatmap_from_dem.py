import os
import numpy as np
import torch
from torchvision import models
import rasterio
from rasterio.windows import Window
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. Paths and Parameters
# ----------------------------------------
COG_PATH = "./05_auxiliary/amazon_biome.tif"
MODEL_PATH = "./02_models/best_resnet.pth"
OUTPUT_HEATMAP_TIFF = "./04_visuals/prediction_heatmap.tif"

LAT_MIN, LAT_MAX = -16, -14.0
LON_MIN, LON_MAX = -66, -64.0

WINDOW_SIZE = 64
STRIDE = 32
BATCH_SIZE = 32
LRM_CLIP = (-20, 20)
SMOOTH_SIGMA = 10.0

# ----------------------------------------
# 2. Locate subregion in DEM
# ----------------------------------------
with rasterio.open(COG_PATH) as src:
    width, height = src.width, src.height
    profile = src.profile.copy()
    row_min, col_min = src.index(LON_MIN, LAT_MAX)
    row_max, col_max = src.index(LON_MAX, LAT_MIN)
    row_start, row_end = sorted((max(0, row_min), min(height, row_max)))
    col_start, col_end = sorted((max(0, col_min), min(width, col_max)))

profile.update(dtype=rasterio.float32, count=1, compress='lzw')

# ----------------------------------------
# 3. Load Model (4-channel ResNet18)
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_state = torch.load(MODEL_PATH, map_location=device)
state = {k.replace('module.', ''): v for k, v in raw_state.items()}

model = models.resnet18(pretrained=False)
model.conv1 = torch.nn.Conv2d(4, model.conv1.out_channels,
                               kernel_size=model.conv1.kernel_size,
                               stride=model.conv1.stride,
                               padding=model.conv1.padding,
                               bias=(model.conv1.bias is not None))

num_classes = state.get('fc.weight', state.get('fc.1.weight')).shape[0]
in_feat = model.fc.in_features
model.fc = torch.nn.Linear(in_feat, num_classes)

mapped = {}
for k, v in state.items():
    if k.startswith('fc.1.'):
        mapped['fc.' + k.split('fc.1.')[1]] = v
    elif not k.startswith('fc.0.'):
        mapped[k] = v

model.load_state_dict(mapped, strict=True)
model.eval().to(device)

# ----------------------------------------
# 4. Feature extraction function (4-channel)
# ----------------------------------------
def compute_features(window_dem):
    dmin, dmax = window_dem.min(), window_dem.max()
    dem_norm = (window_dem - dmin) / (dmax - dmin + 1e-6)

    smooth = ndi.gaussian_filter(window_dem, sigma=SMOOTH_SIGMA)
    lrm = window_dem - smooth
    lo, hi = LRM_CLIP
    lrm = np.clip(lrm, lo, hi)
    lrm = (lrm - lo) / (hi - lo + 1e-6)

    dy, dx = np.gradient(window_dem)
    slope = np.sqrt(dx**2 + dy**2)
    d2y, _ = np.gradient(dy)
    _, d2x = np.gradient(dx)
    curvature = d2x + d2y

    s_lo, s_hi = np.percentile(slope, [1,99])
    slope = np.clip(slope, s_lo, s_hi)
    slope = (slope - s_lo) / (s_hi - s_lo + 1e-6)

    c_lo, c_hi = np.percentile(curvature, [1,99])
    curvature = np.clip(curvature, c_lo, c_hi)
    curvature = (curvature - c_lo) / (c_hi - c_lo + 1e-6)

    return np.stack([dem_norm, lrm, slope, curvature], axis=0)

# ----------------------------------------
# 5. Sliding window inference
# ----------------------------------------
row_indices = list(range(row_start, row_end - WINDOW_SIZE + 1, STRIDE))
col_indices = list(range(col_start, col_end - WINDOW_SIZE + 1, STRIDE))
total_windows = len(row_indices) * len(col_indices)
print(f"Total windows to process: {total_windows}")

results = []
patches, coords = [], []
processed = 0

with rasterio.open(COG_PATH) as src:
    for r in row_indices:
        for c in col_indices:
            dem_win = src.read(1, window=Window(c, r, WINDOW_SIZE, WINDOW_SIZE)).astype(np.float32)
            feat = compute_features(dem_win)
            tensor = torch.from_numpy(feat).unsqueeze(0).float().to(device)
            patches.append(tensor)
            coords.append((r, c))
            processed += 1
            print(f"Processed window {processed}/{total_windows}")
            if len(patches) == BATCH_SIZE:
                batch = torch.cat(patches, dim=0)
                with torch.no_grad():
                    probs = torch.softmax(model(batch), dim=1)[:, 1]
                results.extend([(coords[i][0], coords[i][1], float(p)) for i, p in enumerate(probs.cpu())])
                patches, coords = [], []

    if patches:
        batch = torch.cat(patches, dim=0)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1)[:, 1]
        results.extend([(coords[i][0], coords[i][1], float(p)) for i, p in enumerate(probs.cpu())])

# ----------------------------------------
# 6. Build heatmap over subregion
# ----------------------------------------
sub_H = row_end - row_start
sub_W = col_end - col_start
heatmap_sub = np.zeros((sub_H, sub_W), dtype=np.float32)
count_sub = np.zeros((sub_H, sub_W), dtype=np.int32)

for r, c, p in results:
    rr = r - row_start
    cc = c - col_start
    heatmap_sub[rr:rr+WINDOW_SIZE, cc:cc+WINDOW_SIZE] += p
    count_sub[rr:rr+WINDOW_SIZE, cc:cc+WINDOW_SIZE] += 1

mask = count_sub > 0
heatmap_sub[mask] /= count_sub[mask]

# ----------------------------------------
# 7. Save GeoTIFF and visualize
# ----------------------------------------
new_profile = profile.copy()
new_profile.update({
    'height': sub_H,
    'width': sub_W,
    'transform': rasterio.windows.transform(Window(col_start, row_start, sub_W, sub_H), src.transform)
})

with rasterio.open(OUTPUT_HEATMAP_TIFF, 'w', **new_profile) as dst:
    dst.write(heatmap_sub, 1)

plt.figure(figsize=(10, 8))
plt.imshow(
    heatmap_sub,
    cmap='hot', vmin=0, vmax=1,
    extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    origin='upper'
)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Prediction Heatmap (Sub-region)')
plt.colorbar(label='Probability')
plt.show()
