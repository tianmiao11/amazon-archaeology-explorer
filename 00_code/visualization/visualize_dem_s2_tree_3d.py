import numpy as np
import rasterio
import pyvista as pv
from scipy.ndimage import gaussian_filter, zoom

# === 文件路径 ===
dem_path = r"E:\Users\tiany\Downloads\Final_Archaeological_Site\NASADEM\tif\000000.tif"
s2_path  = r"E:\Users\tiany\Downloads\Final_Archaeological_Site\Sentinel-2\tif\000000.tif"

# === 读取并平滑 DEM ===
with rasterio.open(dem_path) as dem_src:
    dem = dem_src.read(1).astype(np.float32)
    transform = dem_src.transform
dem_smooth = gaussian_filter(dem, sigma=0.5)

# === 读取 S2 并计算 NDVI ===
with rasterio.open(s2_path) as s2_src:
    red = s2_src.read(1).astype(np.float32)    # B4
    nir = s2_src.read(4).astype(np.float32)    # B8
ndvi = (nir - red) / (nir + red + 1e-5)

# ✅ 重采样 NDVI 到 DEM 尺寸
if ndvi.shape != dem.shape:
    zoom_y = dem.shape[0] / ndvi.shape[0]
    zoom_x = dem.shape[1] / ndvi.shape[1]
    ndvi = zoom(ndvi, (zoom_y, zoom_x), order=1)

# === 构建网格 ===
nrows, ncols = dem.shape
x = np.arange(ncols) * transform.a + transform.c
y = np.arange(nrows) * transform.e + transform.f
xx, yy = np.meshgrid(x, y)

# === Z轴高度压缩 ===
z_exaggeration = 0.0001
zz = (dem_smooth - np.nanmin(dem_smooth)) * z_exaggeration

# === 构造地形网格 ===
terrain = pv.StructuredGrid()
terrain.points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
terrain.dimensions = [ncols, nrows, 1]
terrain["Elevation"] = dem_smooth.ravel()

# === 可视化准备 ===
plotter = pv.Plotter()
plotter.add_mesh(terrain, scalars="Elevation", cmap="terrain", show_edges=False)

# === 树的参数 ===
tree_height = 0.0005          # 树高
tree_radius = 0.0005          # 树冠宽度
tree_density_step = 5         # 树的稀疏程度（越大越稀）
tree_opacity = 0.3            # 树的透明度
tree_threshold = 0.8          # NDVI 阈值
tree_indices = np.argwhere(ndvi > tree_threshold)

# === 添加树封装函数 ===
tree_actors = []

def add_trees():
    for i, j in tree_indices[::tree_density_step]:
        if i >= nrows or j >= ncols:
            continue
        x_tree = x[j]
        y_tree = y[i]
        z_tree = zz[i, j] + tree_height * 0.5
        cone = pv.Cone(center=(x_tree, y_tree, z_tree),
                       direction=(0, 0, 1),
                       height=tree_height,
                       radius=tree_radius,
                       resolution=12)
        actor = plotter.add_mesh(cone, color="forestgreen", opacity=tree_opacity)
        tree_actors.append(actor)

# === 控制显示/隐藏状态 ===
tree_enabled = [True]

def toggle_tree(flag):
    tree_enabled[0] = flag
    for actor in tree_actors:
        actor.SetVisibility(flag)
    # 更新文字颜色
    text_actor.SetText(0, "TREE LAYER")
    text_actor.GetTextProperty().SetColor((0, 1, 0) if flag else (1, 0, 0))
    plotter.render()

# === 添加树 & 按钮与文字联动 ===
add_trees()

# 添加提示文字（左下角）
text_actor = plotter.add_text("TREE LAYER", position=(50, 12), font_size=12, color="green")

# 添加复选框按钮
plotter.add_checkbox_button_widget(callback=toggle_tree,
                                   value=True,
                                   position=(10, 10),
                                   size=30,
                                   color_on="green",
                                   color_off="red",
                                   border_size=1)

# === 显示图像 ===
plotter.add_axes()
plotter.show()
