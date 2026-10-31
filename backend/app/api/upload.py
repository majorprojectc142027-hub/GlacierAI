from pathlib import Path
from fastapi import APIRouter, File, UploadFile

from backend.app.ai.segmenter import segment_image
from backend.app.services.file_service import save_uploaded_file

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    """
    Upload a satellite image and run
    U-Net V5 glacier segmentation.
    """

    # Save uploaded image
    file_path = save_uploaded_file(file)

    # Run AI segmentation
    result = segment_image(str(file_path))

    # Return analysis result
    return {
        "filename": Path(file_path).name,
        "status": "Analysis completed successfully",
        "model": "U-Net V5",
        "glacier_pixels": result["glacier_pixels"],
        "total_pixels": result["total_pixels"],
        "glacier_coverage": round(
            result["glacier_percentage"], 2
        ),
    }
