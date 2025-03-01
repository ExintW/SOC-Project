import rasterio
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

# 📂 输入与输出文件路径
tiff_path = r"D:\EcoSci\Dr.Shi\Data\wammaogou_landuse\wammaogou_reprojected.tif" # 已重新投影的 TIFF
output_shp = r"D:\EcoSci\Dr.Shi\Data\wammaogou_landuse\wammaogou_boundary.shp"   # 输出 Shapefile


# 1️⃣ 读取 TIFF 文件并提取矢量
with rasterio.open(tiff_path) as src:
    raster = src.read(1)
    mask = raster != src.nodata  # 生成掩膜排除无效值
    transform = src.transform
    crs = src.crs

    # 提取所有矢量轮廓
    shapes_generator = shapes(raster, mask=mask, transform=transform)
    geometries = [shape(geom) for geom, value in shapes_generator]

# 2️⃣ 合并所有多边形为单一多边形 (提取最外圈边界)
merged_polygon = unary_union(geometries)  # 合并为单一外轮廓

# 3️⃣ 转换为 GeoDataFrame 并保存
gdf = gpd.GeoDataFrame([{"geometry": merged_polygon}], crs=crs)
gdf.to_file(output_shp, driver='ESRI Shapefile')

print(f"✅ 最外圈边界已提取并保存至: {output_shp}")
