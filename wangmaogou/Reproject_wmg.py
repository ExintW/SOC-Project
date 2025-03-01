import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 📝 设置支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号

# 📂 文件路径
tiff_path = r"D:\EcoSci\Dr.Shi\Data\wammaogou_landuse\王茂沟水利生态地理要素20171.tif"
reprojected_tiff = r"D:\EcoSci\Dr.Shi\Data\wammaogou_landuse\wammaogou_reprojected.tif"

# 🎯 目标坐标系 WGS84 (EPSG:4326)
dst_crs = 'EPSG:4326'

# 1️⃣ 重新投影 TIFF 文件
with rasterio.open(tiff_path) as src:
    print(f"原始坐标系: {src.crs}")

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # 保存为新的 TIFF 文件
    with rasterio.open(reprojected_tiff, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
    print(f"✅ 已重新投影并保存至: {reprojected_tiff}")

# 2️⃣ 绘制重新投影后的 TIFF 并修复坐标轴问题
with rasterio.open(reprojected_tiff) as src:
    print(f"✅ 重新投影后的坐标系: {src.crs}")

    # ✅ 获取转换后的经纬度边界
    geo_bounds = transform_bounds(src.crs, dst_crs, *src.bounds)
    print(f"✅ 经纬度范围: {geo_bounds}")

    fig, ax = plt.subplots(figsize=(12, 10))

    # ✅ 使用 extent=geo_bounds 修复坐标轴问题
    show((src, 1), ax=ax, cmap='terrain', extent=geo_bounds)

    # 🎨 设置标题与坐标轴标签（支持中文）
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='x')  # 设置 x 轴为普通数字显示
    ax.set_title("王茂沟 TIFF （正确经纬度）", fontsize=16)
    ax.set_xlabel("经度", fontsize=12)
    ax.set_ylabel("纬度", fontsize=12)
    plt.show()
