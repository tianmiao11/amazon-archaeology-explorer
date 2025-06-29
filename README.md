# 1. 项目简介与背景（Introduction & Background）

## 🌍 探索亚马逊：用 AI 追寻遗失的文明

亚马逊热带雨林横跨南美九国，覆盖超过 600 万平方公里，长久以来被视为“地球之肺”与人类最后的疆域之一。然而，在郁郁葱葱的森林深处，越来越多的证据表明，这片土地并非如传统所认为的“人类未曾涉足之地”，而是孕育过复杂、发达的史前文明。

如今，借助高分辨率遥感影像、LIDAR 激光雷达和人工智能工具，我们有机会揭开隐藏在林冠之下的文明遗迹。

OpenAI 发起的 **“To Z Challenge”** 鼓励全球研究者利用 o3、o4 mini 和 GPT-4.1 等大模型，结合开源遥感与历史数据，自主发现潜在的考古遗址，为探索“Z 城”“Paititi”“El Dorado”等传说提供数字证据，推动公众参与亚马逊历史重建。

---

## 📌 本项目目标与方法总览

在本项目中，我们以“系统性发现亚马逊地区未知遗迹点”为目标，构建了一个端到端的 AI 考古推理流程，包括：

1. **数据构建**：整合六篇公开考古研究论文中提取的坐标点（共计 4607 个正样本），在全亚马逊范围内生成等量、均匀分布、2 公里避让的负样本点，建立训练所需的数据集。
2. **模型训练**：
   - 基于 NASADEM 高程数据，提取地形五通道特征（原始高程、平滑、高斯坡度、拉普拉斯边缘、局部标准差），训练 ResNet 模型，识别可能的人工地貌痕迹。
   - 使用 Sentinel-2 RGB 图像训练另一个模型，识别林下结构异常、视觉 speckle 斑点、植被纹理等潜在线索。
3. **区域推理与聚类整合**：在全亚马逊范围滑窗推理，分别从两种模型中提取 Top 1000 高分候选点，使用空间聚类（DBSCAN）与空间近邻合并生成最终候选遗址点。
4. **空间过滤与分析**：针对 DEM 结果，剔除靠近河流（<2km）区域；针对 S2 结果，过滤靠近已知城镇干扰区，确保候选点更可能为未被发现的古迹。
5. **可视化与人机交互平台**：设计开发一个网页地图工具，用户可在界面上自由框选区域或输入坐标，直观查看该区域内所有潜在遗迹预测结果，便于后续专家核查或实地调查。

---

## 🔍 项目意义

本项目不仅是一次技术挑战，更是一次数据时代的考古探索尝试。它展示了：

- 开源遥感数据与 AI 模型在大尺度文化遗产发现中的潜力；
- 平民科学家也能参与遗迹定位、史前文明研究；
- 基于 reproducible pipeline 的空间智能分析如何支持更可靠、更透明的学术发现。

我们希望通过本项目，推动亚马逊地区的历史认知向前一步，也为全球其他森林地区的考古研究提供方法借鉴。

数据集说明
- [**NASADEM: NASA 30m Digital Elevation Model**](https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001): A 30-meter resolution DEM provided by NASA for global elevation mapping.
- [**Harmonized Sentinel-2 MSI: Level-2A Surface Reflectance**](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED): Harmonized Sentinel-2 imagery with surface reflectance (L2A), optimized for analysis and machine learning.
- [**MERIT Hydro: Global Hydrography Datasets**](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1): High-accuracy global hydrography data including rivers, flow direction, and basins.


# 2. 数据构建与预处理（Data Collection & Preprocessing）

为确保训练数据质量与考古学意义兼具，我们从六篇权威考古研究中提取坐标点作为正样本，并构建等量的负样本，覆盖整个亚马逊生物群区。随后结合遥感数据（NASADEM 与 Sentinel-2），提取训练模型所需的地形与光谱特征。

---

## ✅ 正样本采集（Positive Samples）

我们从以下六篇公开论文中提取出考古遗址坐标，共计 4607 个正样本点：


| 序号 | 文献标题 | 样本数量 | 样本描述 |
|------|----------------------------------------------------------------------------------------------------------------------------------------|------------|-------------------------|
| 1 | [*More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia*](https://www.science.org/doi/10.1126/science.ade2541) | 1181       | earthwork              |
| 2 | [*Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning*](https://peerj.com/articles/15137) | 1811       | earthwork / ADE / other |
| 3 | [*Geometry by Design: Contribution of Lidar to the Understanding of Settlement Patterns of the Mound Villages in SW Amazonia*](https://journal.caa-international.org/articles/10.5334/jcaa.45) | 41         | earthwork              |
| 4 | [*Geolocation of unpublished archaeological sites in the Peruvian Amazon*](https://www.nature.com/articles/s41597-021-01067-7)           | 307        | earthwork              |
| 5 | [*Hundreds of Geoglyphs Discovered in the Amazon*](https://jqjacobs.net/archaeology/geoglyph.html)                                       | 1118       | earthwork (geoglyphs)  |
| 6 | [*Lidar reveals pre-Hispanic low-density urbanism in the Bolivian Amazon*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9177426)         | 149        | earthwork (mound)      |



这些坐标涵盖巴西、秘鲁、玻利维亚等地区，地理分布广泛，类型丰富（包括几何土堤、平台、mound 聚落、道路网等），为模型提供了多样的正样本。
样本大小采用2km*2km左右。因为根据文章6的结果，遗迹在这个范围内可以较好的展示（约处于中心75%的区域内）
数据处理：将所有2km内的遗迹点进行合并，避免重复出现

---

## ❎ 负样本生成（Negative Samples）

为构建对照数据集，我们在亚马逊全境内生成 4607 个负样本点，确保以下条件：

- **范围约束**：全部落在 Amazon Biome 边界内（基于公开 GeoJSON）
- **距离约束**：每个负样本距离任意正样本 ≥ 2 km，避免采样到已知遗址周边
- **分布均匀**：采用网格划分 + 分区随机采样策略，确保点位覆盖均衡，避免模型学习区域偏差

生成方式完全可复现，支持地理空间分析工具（如 Shapely、GeoPandas）进行验证与可视化。

---

## 🛰️ 遥感数据获取（Remote Sensing Data Acquisition）

为提取模型训练所需的空间特征，我们从公开遥感数据源中获取了每个点位的地形和多光谱影像：

### 1. **NASADEM（地形数据）**
- 来源：**NASA / NASADEM_HGT/001**（Google Earth Engine 数据集）
- 空间分辨率：约 30 米
- 提取方式：
  - 每个样本点为中心裁剪出 **64×64 像素 patch**（约 1920m × 1920m）
  - 导出原始高程值（Raw Elevation）
- 注：地形导数特征（如平滑、坡度、拉普拉斯、局部标准差）**在模型训练阶段实时计算**，并未保存在初始数据中

### 2. **Sentinel-2（多光谱数据）**
- 来源：**COPERNICUS/S2_SR_HARMONIZED**（通过 Google Earth Engine 访问）
- 波段使用：
  - B4（红）/B3（绿）/B2（蓝） → 用于 RGB 可视化与 CNN 输入
  - B8（近红外） → 后续用于计算 NDVI speckled 掩膜
- 图像处理流程：
  - 时间范围限定为 **2024 年全年**
  - 过滤云量（`CLOUDY_PIXEL_PERCENTAGE < 10%`）
  - 使用 `.mosaic()` 合成全年影像
  - 裁剪为 **256×256 像素 patch**（对应地面范围约 2560m × 2560m）
  - 所有空值和云区自动掩膜处理，提升数据质量

---

## 📦 数据结构与组织（Data Structure & Format）

最终每个样本点的存储结构如下：

- `lat, lon`：中心坐标（WGS84）
- `type`：类别标签（正样本 / 负样本）
- `dem_patch`：原始高程图像，shape: `[1, 64, 64]`
- `s2_rgbn`：多光谱图像，包含 B4, B3, B2, B8，shape: `[4, 256, 256]`

说明：
- `.tif` 格式用于高效读取与 GPU 批量加载
- 地形导数特征（如坡度、边缘）在模型训练数据管道中即时计算，确保灵活性与复用性
- 所有 patch 保持中心对齐，具备一致的空间参考

---

## 📎 数据质量控制措施

- 对正样本进行人工去重与地理空间清洗
- 对负样本进行空间约束验证与可视化检查
- 所有遥感图像均进行异常值剔除、尺寸归一化与归一化处理
- Patch 自动居中裁剪，适配 CNN 网络输入结构

---

通过上述数据构建流程，我们获得了一个平衡、空间合理、特征丰富的考古点数据集，能够有效支持地形/纹理双通道模型的训练与泛化。

# 3. 模型训练与推理流程（Model Training & Inference Pipeline）

为深入挖掘地形与植被纹理中的潜在线索，我们分别基于 NASADEM 和 Sentinel-2 图像，构建了两个独立的分类模型，用于学习正/负样本之间的遥感特征差异。最终，我们将两个模型的输出进行空间聚类与融合，从而筛选出最有可能的新遗址候选点。

---

## 🛰️ 3.1 DEM 模型（NASADEM）

### 📐 输入特征构建（5 通道 DEM stack）

以每个坐标点为中心，从 NASADEM 中提取 64×64 像素的地形 patch（分辨率约 1920m × 1920m），并构造以下 5 个通道特征作为模型输入：

1. **原始高程（Raw Elevation）**  
   将原始高程值进行归一化，公式如下：  
   `E_norm(x, y) = (E(x, y) - E_min) / (E_max - E_min + 1e-6)`

2. **高斯平滑（Gaussian Smoothing）**  
   使用标准差 σ=10 进行高斯模糊处理：  
   `G(x, y) = GaussianFilter(E_norm, sigma=10)`

3. **坡度（Slope Magnitude）**  
   首先计算 x、y 方向上的梯度，再合成坡度幅值：  
   `grad_x, grad_y = np.gradient(G)`  
   `S(x, y) = sqrt(grad_x² + grad_y²)`

4. **拉普拉斯边缘（Laplacian）**  
   使用二维拉普拉斯算子增强边缘：  
   `L(x, y) = laplace(G)`

5. **局部标准差（Local Std Dev）**  
   以 5×5 滑窗计算局部标准差，公式如下：  
   `mean = uniform_filter(G, size=5)`  
   `mean_sq = uniform_filter(G², size=5)`  
   `std_local = sqrt(max(0, mean_sq - mean²))`

所有通道最终被堆叠为一个 `[5, 64, 64]` 的输入 patch，并对每个通道分别执行标准化：

`X'_i = (X_i - mean_i) / (std_i + 1e-6)`

---

### 🧠 模型结构与训练策略

- 主干网络：**ResNet18**
- 修改：支持 5 通道输入
- 损失函数：二分类交叉熵（Binary Cross-Entropy）
- 优化器：Adam
- 学习率调度：ReduceLROnPlateau（基于验证准确率）
- 数据增强：旋转、翻转、光照扰动等基础图像增强操作

在训练集中，模型快速收敛并在验证集上达到了约 **70% 的准确率**，表现出较强的人工地形识别能力，尤其对微弱的边缘、高差区域反应明显。

## 🌿 3.2 Sentinel-2 模型（S2）

### 🎨 输入特征

输入由 Sentinel-2 图像中提取的 RGB 三通道 patch 及 NDVI-based speckle mask 构成，合计 **5 个通道**（`[5, 206, 204]`）。  
其中 **speckle mask** 专为识别“林下斑驳区块”而构建，指的是在 **高 NDVI 区域（NDVI > 0.7）** 中出现的灰度纹理异常区域，可能暗示林地下存在非自然结构或地表扰动。

### 🧠 模型结构与训练策略

本模型采用轻量级 **ResNet-18** 架构，并将输入通道数由 3 扩展至 5，以融合 NDVI speckle 信息。整体训练策略如下：

- **增强策略**：使用 Center Crop 对图像进行中心裁剪，统一 patch 尺寸；
- **损失函数**：CrossEntropyLoss；
- **优化器与调度器**：Adam 优化器，结合 ReduceLROnPlateau 学习率调度；
- **样本构建**：
  - 正样本来自真实考古点；
  - 负样本从远离正样本 >2km 且分布均匀区域中采样。

模型训练目标是识别人类活动在林地中的纹理扰动，例如：

- 树冠边界的几何化趋势；
- 林下空间中 speckle 区域的异常分布；
- 色彩与灰度的跳变与碎裂感。

最终模型在验证集上达到约 **75% 的准确率**，在潜在遗迹区域的识别中展现出较强的结构感知与异常检测能力。

## 🌍 3.3 全图推理与候选点提取（Sliding Window Inference & Clustering）

为了在 Amazon Biome 范围内全面发现潜在遗迹，我们使用模型进行滑窗推理生成预测热图，并基于概率高的区域执行聚类，最终提取出 Top-K 个最有可能的候选遗址。

---

### 📊 滑窗推理（Sliding Window Inference）

我们将整个遥感影像划分为小块区域，使用训练好的模型对每个 patch 进行预测，生成概率热图。核心步骤如下：

```text
输入影像：DEM（COG 格式）或 Sentinel-2 影像
DEM 分辨率：30 米，窗口大小设置为 64 像素（1920 米）
S2 分辨率：10 米，窗口大小设置为 256 像素（2560 米）
滑动步长（stride）：通常为窗口宽度的 1/2，用于增强覆盖率
每批推理大小（batch）：32 个 patch
输出：每个 patch 的“遗迹概率”（0~1）
```

每个窗口 patch 都从原始 DEM 中动态计算 4 通道特征：

```text
1. 高程归一化（dem_norm）：
   dem_norm = (DEM - min) / (max - min)

2. 局部高程残差（lrm）：
   smooth = GaussianFilter(DEM, sigma=10)
   lrm = clip((DEM - smooth), -20, 20)

3. 坡度（slope）：
   dx, dy = np.gradient(DEM)
   slope = sqrt(dx^2 + dy^2)

4. 曲率（curvature）：
   d2x, d2y = np.gradient(dx), np.gradient(dy)
   curvature = d2x + d2y
```

模型输出为每个 patch 的“遗迹概率”，所有窗口结果汇总后，按像素加权平均，构成预测热图（Heatmap）。热图使用 GeoTIFF 格式保存，并在地理坐标下可视化展示。

---

### 🧪 空间聚类与模型融合（DBSCAN Clustering on Heatmap）

对滑窗生成的热图，我们采用 DBSCAN 对概率高于阈值的区域进行空间聚类，提取出置信度高、密度聚集的候选点。主要流程如下：

```text
热图来源：模型输出概率图（GeoTIFF 格式）
聚类方法：DBSCAN（密度聚类）
处理单位：Tile 分块（每块 2048 × 2048 像素，防止内存爆炸）
```

#### 📌 核心参数解释：

```text
prob_threshold = 0.9     # 参与聚类的最小概率
eps_pixels     = 32      # DBSCAN 空间邻近半径（单位：像素）
min_samples    = 3       # 构成有效聚类的最少点数
top_k_clusters = 1000    # 选取得分最高的前 K 个聚类中心
```

#### 📌 聚类提取逻辑：

```text
1. 在每个 tile 中提取概率 > 0.9 的像素点作为候选点
2. 使用 DBSCAN 聚类找出空间聚集区域
3. 对每个聚类提取其：
   - 聚类点数 n_points
   - 最大概率 prob_max
   - 平均概率 prob_mean
   - 最大概率位置作为中心点（row, col → lat, lon）
4. 所有聚类中心按 prob_max 排序，取 Top-K 个保存
```

最终输出一个包含所有高置信聚类中心的 CSV 文件，每行记录如下字段：

```text
- lon, lat         候选点的经纬度
- prob_max         聚类中最大预测概率
- prob_mean        聚类中平均预测概率
- n_points         聚类内像素点数量
```

---

这一流程确保了模型输出与实际空间尺度的一致性，同时过滤了单点噪声，通过密度增强方式选出“值得进一步分析的区域”，为最终候选遗迹点生成提供高质量输入。

## 🔗 DEM 与 S2 候选点匹配说明（Matching DEM and Sentinel-2 Clusters）

为提升预测置信度，我们将 DEM 模型与 Sentinel-2 模型的高概率聚类结果进行空间匹配，筛选两者都预测为“可能存在遗迹”的交集点。

### ✅ 匹配规则

- 输入文件：
  - `cluster_nasadem.csv`（DEM 模型结果）
  - `cluster_sentinel2.csv`（S2 模型结果）

- 匹配条件：
  1. **地理距离 < 1 km**：DEM 点与 S2 点之间的地理距离小于 1 公里；
  2. **一对一匹配**：每个 DEM 点最多匹配一个 S2 点；
  3. **输出字段**：
     - `lon`, `lat`：坐标
     - `prob_dem`：DEM 预测概率
     - `prob_s2`：S2 预测概率

- 输出文件：`matched_clusters.csv`（包含所有成功匹配的候选点）

### 🚫 后续过滤建议（正样本排除）

为确保发现为“新遗迹”，应从匹配结果中排除所有**距离正样本 < 1 km**的点。该逻辑可通过加载正样本坐标并使用空间距离计算实现。


## 🧽 3.4 空间过滤策略

为了排除自然因素（如河流冲积地）和现代活动（如城市化区域）对模型结果的干扰，我们对最终候选点集引入以下空间层级的过滤与增强处理：

### 🚫 空间过滤策略

| 模型来源 | 空间过滤策略说明 |
|----------|------------------|
| **DEM 模型** | 剔除所有 **距离河流中心线小于 2 公里** 的候选点，以规避河漫滩沉积、冲沟等地形在高程图上的误判干扰 |
| **S2 模型** | 剔除靠近现代城市或植被严重破坏区的候选点（通过手动黑名单或夜光图辅助实现） |

---

这些空间过滤与增强策略，有效提升了最终候选点的可信度，减少了因自然或人为因素造成的误报风险，也为后续专家复核提供更具针对性的目标点。

# 4. 候选点分析与成果展示（Candidate Site Analysis & Key Findings）

在完成对 Amazon Biome 区域的模型推理与空间聚类整合后，我们最终获得了一组**高度可信的潜在遗址候选点**。本节对这些候选点的整体特征、分布情况及若干代表性点位进行分析，并展示关键图像样例，以支持人工验证与后续研究。

---

## 📍 4.1 候选点分布概览

### 📊 数量与来源
- **初步模型推理结果**：
  - DEM 模型 Top 1000 高得分点
  - S2 模型 Top 1000 高得分点
- **融合结果**：
  - 匹配点对数量（两模型距离 < 1 km）：**约 500 个**
  - 经过空间过滤（去除河流、城镇等干扰）后保留：**约 350 个最终候选点**

这些候选点具有以下空间分布特征：
- 高密度聚集区主要分布在巴西亚马逊中部、玻利维亚北部与秘鲁东部
- 多数点位处于偏远森林边缘或河流支流上方缓坡台地，贴合已知史前聚落选址逻辑
- 部分候选点接近历史记录中的探险路线，如库希库古以南、马德雷德迪奥斯河以北一带

---

## 🌄 4.2 候选点特征分析

### 📐 地形（DEM）表现特征：
- 高程变化缓慢但存在**矩形、弧形、圆形轮廓**
- 局部坡度异常明显，具备“低洼中环”“中央高突”模式
- 拉普拉斯边缘图显示出连续边界，与典型 earthwork 或平台聚落相符

### 🌿 光学（S2）表现特征：
- 林冠色彩 speckled，可能反映密林下人工清理或次生植被回生
- 存在线性或几何边缘区域，颜色突变与周边形成视觉边界
- 有些 patch 中存在明显植被间隙，与人工结构上方植被退化模式一致

---

## 🔍 4.3 代表性候选点示例

以下展示几个代表性点位，包含其基本信息与 patch 图像分析：

### 📌 候选点 #112

- **坐标**：`-7.2451, -63.0984`
- **区域**：靠近罗亚尔马亚河支流，接近地势台地
- **DEM 特征**：
  - 高程图显示出规则椭圆状隆起
  - 坡度图形成环状缓坡结构
- **S2 特征**：
  - RGB 显示出中心植被变异 speckle pattern
  - 四周林冠呈现交错对称清理痕迹

### 📌 候选点 #207

- **坐标**：`-11.5123, -68.0247`
- **区域**：秘鲁东部雨林边缘，靠近已知历史探险路线
- **DEM 特征**：
  - 中心高凸区与周围低洼区形成“岛状”视觉结构
- **S2 特征**：
  - 存在淡色植被斑块与模糊直线边缘，可能对应古道或围墙基础

### 📌 候选点 #330

- **坐标**：`-10.9217, -61.3098`
- **区域**：接近 Acre 州边界，与已知 geoglyph 聚集区相距约 80 公里
- **DEM 特征**：
  - 显示矩形边框与对角轴对称形态
- **S2 特征**：
  - 植被色彩分布高度不均匀，具有人为干预区域特征

---

## ✍️ 4.4 人工验证建议

基于目前的空间、图像和模型信号结果，我们推荐以下几类候选点优先考虑进入后续验证流程：

- **结构明显**：地形呈现几何封闭形状的点（如圆形或矩形平台）
- **多模态共识**：同时在 DEM 与 S2 中显示高得分、清晰纹理边界的点
- **独立孤立点**：远离已知考古遗址聚集区但具备结构特征的偏远新点

人工验证建议方式包括：
- 高分辨率图像比对（如 Google Earth Pro 历史图层）
- 对照历史地图与文献描述（如传说中路线/山脊/高台）
- 联合专家或志愿考古学者初步远程辨识

---

## ✅ 小结

通过模型双通道交叉验证、空间过滤和图像人工分析，我们从百万级遥感 patch 中挖掘出一批潜在的遗迹候选点。这些点在地形、纹理、空间聚集等维度上均呈现显著特征，为后续学术研究与实地勘探提供了可操作的起点。

下一步，我们将结合这些点，开发可视化交互平台，支持在线地图浏览、坐标定位与专家协作分析。

# 5. 可视化与交互平台设计（Visualization & Interactive Platform）

为了提升候选遗迹点的可解释性与展示效果，并促进专家与公众的参与，我们设计开发了一个基于地图的交互式可视化平台。该平台不仅能直观呈现模型输出结果，还支持用户自定义区域查询、图像查看与数据导出等功能，是本项目成果的关键组成部分之一。

---

## 🗺️ 5.1 平台目标与意义

- **成果验证**：提供直观界面，便于专家通过地图点击、图像比对快速判断候选点的可疑性
- **区域探索**：用户可自由框选任意区域查看潜在遗址密度与特征分布
- **交互共享**：为考古学者、研究人员或公众科普者提供开放入口，促进合作交流
- **追踪更新**：支持未来模型优化后快速更新结果，形成持续演进的考古预测地图

## 💻 5.2 核心功能模块

### 🗺️ 地图视图与候选点展示

- **底图支持**：用户可切换卫星图、地形图等背景图层
- **候选点图层管理**：
  - 可独立切换显示 DEM 模型结果、Sentinel-2 模型结果及融合结果
  - 所有点位以交互式红点呈现，点击或悬停后显示详细信息
- **点位信息展示**：
  - 经纬度、模型评分（DEM & S2）
  - DEM 图像 Patch（高程 / 坡度 / 边缘）
  - Sentinel-2 RGB Patch
  - 可视化 3D GLB 地形模型，支持切换 “Terrain Only” 与 “With Tree” 视图
  - **GPT-4o 考古解释内容**（详见下文）

### 🔎 区域搜索与列表检索

- 地图支持自由框选任意区域，自动高亮该范围内的候选点
- 侧边栏可折叠显示所有点的编号（ID），点击跳转定位并展开详情
- 也可通过输入中心坐标 + 半径快速定位

### 📤 数据导出与引用支持

- 可导出当前视图范围内候选点的：
  - 坐标列表（CSV / GeoJSON）
  - Patch 图像（PNG 格式）
- 每个点绑定唯一编号（NO），便于跨平台引用或后续标注

---

## ⚙️ 5.3 技术实现要点

> *本节简述核心技术，重点突出“可复现性”和“快速交互”*

- **前端框架**：使用 [Leaflet.js](https://leafletjs.com) 构建交互地图界面，支持 GeoJSON 动态加载与响应式操作
- **三维模型展示**：引入 `<model-viewer>` 实现 Web 原生 3D 可视化，支持 Terrain / Tree 模型切换
- **图像交互展示**：点击可放大任一 DEM / S2 Patch，支持滚轮缩放查看细节
- **后端支持**：由轻量 Python 脚本（如 Flask / FastAPI）支撑 Patch 图像加载；所有结果本地预生成，避免动态延迟

---

## 🧪 5.4 示例交互演示（可选附图）

- ✅ 示例 1：点击点位 `#001123`，查看其 DEM / S2 Patch 与 3D 地形
- ✅ 示例 2：框选马亚西部流域区域，筛选其中高置信度候选点
- ✅ 示例 3：切换 NDVI 图层，观察斑驳 speckle 区域与地形平台的一致性

---

## 🧠 5.5 GPT-4o 考古解释生成（每个候选点）

平台为每个候选点自动生成结构化的考古解读内容，结合遥感图像与地形分析，帮助用户高效理解潜在遗迹的空间特征与考古意义：

1. **表面分析（Sentinel-2）**  
   识别植被突变、几何清理痕迹、斑驳分布等非自然图像特征，判断是否存在人类干预。

2. **高程分析（NASADEM）**  
   检测局部平台、微地形隆起、对称沟槽等可能代表台地、墓冢、排水结构的特征。

3. **空间一致性分析**  
   分析 Sentinel-2 图像与 DEM 高程在几何位置上的匹配情况，强化遗迹解释可信度。

4. **考古解释与结论**  
   推断该点是否为可能的聚落遗址、农业场地或仪式空间，参考亚马逊地区已知文化类型。

5. **后续建议**  
   根据图像特征推荐后续工作，如高分辨率 LiDAR 扫描、无人机成像、实地考古调查等。

- 所有内容由 GPT-4o 基于 DEM + Sentinel-2 图像自动生成，并与候选点绑定展示，提升点位解读效率与可复现性。

---

## ✨ 5.6 平台潜力与扩展计划

- **专家协作验证**：将支持用户登录并对候选点打分、标记“疑似等级”，形成专家共识图层
- **接入高分影像**：探索接入 PlanetScope、Maxar、历史 Google Earth 图层等，辅助高分辨率验证
- **模型接口扩展**：支持上传自定义模型预测结果，与现有平台结果对比叠加

---

## ✅ 小结

本平台不仅是“AI 发现亚马逊遗迹”的展示界面，更是促进遥感学者、考古专家与 AI 工程师协作的桥梁。  
通过开放的数据交互、结构化展示与三维可视化能力，推动考古研究真正“落地可视”，走出论文、走向地图。

## 👁️ 6 人工筛选与遗迹确认（Visual Inspection & Final Judgement）

在模型完成推理、空间聚类和多层过滤后，我们获取了一个融合 DEM 与 Sentinel-2 模型的高置信度候选遗迹点集。然而，自动模型依然可能受限于训练数据偏差、遥感干扰等因素。因此，我们引入人工视觉审核作为最后一环：

### ✅ 人工筛选流程

1. **多图对照**：
   - 分别查看每个候选点的 DEM 渲染图（地形起伏）与 S2 RGB 图（地表纹理）
   - 对比其空间结构特征、几何形态与已知遗迹相似度

2. **基于考古“常识”的初步判断**：
   - 是否具备规则形状（圆形、方形、线状排布等）？
   - 是否有成簇或线性分布、道路/水道连接等特征？
   - 是否位于常见的遗迹选址地形，如河岸台地、开阔平地？

3. **综合判断打分**：
   - 每个候选点按照视觉显著性与结构置信度进行打分
   - 选出一批最具代表性的 **Top K 最终候选点**，用于最终展示与后续验证

### 🧩 示意图例（略）

本环节强化了模型输出的可解释性，借助人类经验为自动识别“兜底”，确保提交的遗迹点具备考古逻辑合理性与研究探索价值。


Reference
[1] de Souza, J. G. (2014). Nutrient Cycling in the Amazon: A Review of the High Decomposition Rates and Soil Fertility Implications. SIT Graduate Institute. Retrieved from https://digitalcollections.sit.edu/isp_collection/2339

[2] Khan, S., Aragão, L., & Iriarte, J. (2017). A UAV–lidar system to map Amazonian rainforest and its ancient landscape transformations. International Journal of Remote Sensing, 38(8–10), 2313–2330. https://www.tandfonline.com/doi/abs/10.1080/01431161.2017.1295486

[3] Robert S. Walker, Jeffrey R. Ferguson, Angelica Olmeda, Marcus J. Hamilton, Jim Elghammer & Briggs Buchanan. (2023) Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning. PeerJ 11, pages e15137. https://peerj.com/articles/15137/

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778. https://doi.org/10.1109/CVPR.2016.90

[5] Prümers, H., Robinson, M., Alves, D., et al. (2023). More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia. Science, 379(6637), 1347–1351. https://doi.org/10.1126/science.ade2541

[6] Levis, C., Silva, T. S. F., Costa, F. R. C., et al. (2023). Predicting the geographic distribution of ancient Amazonian archaeological sites with machine learning. PeerJ, 11, e15137. https://doi.org/10.7717/peerj.15137

[7] Hesse, R., & Moraes, C. D. P. (2021). Geometry by design: Contribution of lidar to the understanding of settlement patterns of the mound villages in SW Amazonia. Journal of Computer Applications in Archaeology, 4(1), 61–76. https://doi.org/10.5334/jcaa.45

[8] Watling, J., Iriarte, J., Robinson, M., et al. (2021). Geolocation of unpublished archaeological sites in the Peruvian Amazon. Scientific Data, 8, Article 157. https://doi.org/10.1038/s41597-021-01067-7

[9] Jacobs, J. Q. (n.d.). Hundreds of Geoglyphs Discovered in the Amazon. Retrieved from https://jqjacobs.net/archaeology/geoglyph.html

[10] Prümers, H., Robinson, M., Mologni, F., et al. (2022). Lidar reveals pre-Hispanic low-density urbanism in the Bolivian Amazon. Nature, 606(7912), 325–328. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9177426