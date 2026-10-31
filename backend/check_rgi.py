
import geopandas as gpd
import rasterio
from shapely.geometry import box
from app.services.rgi_service import RGI_FILE

glacier_id = "RGI2000-v7.0-G-15-00467"
image_path = r"D:\GlacierAI\datasets\live\rgi2000_v7_0_g_15_00467_latest.tif"

gdf = gpd.read_file(RGI_FILE)
gdf = gdf[gdf["rgi_id"].astype(str) == glacier_id].to_crs("EPSG:32644")

with rasterio.open(image_path) as src:
    glacier = gdf.geometry.iloc[0]
    image_footprint = box(*src.bounds)

    total_area = glacier.area
    covered_area = glacier.intersection(image_footprint).area

    print("Total RGI polygon area (km²):", round(total_area / 1e6, 3))
    print("RGI area inside image (km²):", round(covered_area / 1e6, 3))
    print("Polygon coverage (%):", round(covered_area / total_area * 100, 2))
