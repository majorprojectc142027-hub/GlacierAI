
from pathlib import Path
import geopandas as gpd

path = Path(r"D:\GlacierAI\datasets\processed\global_rgi.gpkg")
glacier_id = "RGI2000-v7.0-G-15-00467"

gdf = gpd.read_file(path)

print("Inventory CRS:", gdf.crs)
print("Available columns:", list(gdf.columns))

matches = gdf[
    gdf["rgi_id"].astype(str).str.strip() == glacier_id
]

print("\nMatching records:", len(matches))

if matches.empty:
    print("ERROR: Gangotri RGI ID was not found.")
else:
    row = matches.iloc[0]
    print("RGI ID:", row["rgi_id"])
    print("Geometry type:", row.geometry.geom_type)
    print("Geometry valid:", row.geometry.is_valid)
    print("Geometry bounds:", row.geometry.bounds)
    print("Geometry area in inventory CRS:", row.geometry.area)
