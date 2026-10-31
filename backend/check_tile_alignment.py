
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

ROOT = Path(r"D:\GlacierAI")
DATA = ROOT / "datasets" / "live"
PRED = DATA / "predictions"
RGI = ROOT / "datasets" / "processed" / "global_rgi.gpkg"

GLACIER_ID = "RGI2000-v7.0-G-15-00467"
PREFIX = "rgi2000_v7.0_g_15_00467_tile_"

inventory = gpd.read_file(RGI)
glacier = inventory[
    inventory["rgi_id"].astype(str).str.strip() == GLACIER_ID
]

if glacier.empty:
    raise SystemExit(f"Glacier not found: {GLACIER_ID}")

for mask_path in sorted(PRED.glob(f"{PREFIX}*_glacier_mask.tif")):
    image_path = DATA / mask_path.name.replace(
        "_glacier_mask.tif", ".tif"
    )

    print("\n" + "=" * 60)
    print("Tile:", mask_path.name)

    with rasterio.open(mask_path) as mask_src:
        predicted = mask_src.read(1) == 1
        mask_crs = mask_src.crs
        mask_transform = mask_src.transform
        mask_bounds = mask_src.bounds

        print("Mask shape:", predicted.shape)
        print("Mask CRS:", mask_crs)
        print("Mask bounds:", mask_bounds)
        print("Predicted pixels:", int(predicted.sum()))

    if not image_path.exists():
        print("Matching source image not found:", image_path.name)
    else:
        with rasterio.open(image_path) as image_src:
            print("Source shape:", (image_src.height, image_src.width))
            print("Source CRS:", image_src.crs)
            print("Source bounds:", image_src.bounds)
            print(
                "Same grid:",
                image_src.shape == predicted.shape
                and image_src.crs == mask_crs
                and image_src.transform == mask_transform
            )

    projected = glacier.to_crs(mask_crs)

    reference = rasterize(
        [
            (geom, 1)
            for geom in projected.geometry
            if geom is not None and not geom.is_empty
        ],
        out_shape=predicted.shape,
        transform=mask_transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    intersection = int(np.count_nonzero(predicted & reference))
    union = int(np.count_nonzero(predicted | reference))
    iou = intersection / union if union else 0.0

    print("RGI reference pixels:", int(reference.sum()))
    print("Overlap pixels:", intersection)
    print(f"IoU: {iou:.4f} ({iou * 100:.2f}%)")
