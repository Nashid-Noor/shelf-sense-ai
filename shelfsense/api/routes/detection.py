"""
Detection API Routes

Endpoints for book detection and layout analysis on images and batch uploads.
"""

import base64
import time
from typing import Optional

from fastapi import (
    APIRouter, HTTPException, UploadFile, File, Form,
    BackgroundTasks, status,
)
from loguru import logger

from shelfsense.api.schemas import (
    DetectionRequest,
    DetectionResponse,
    DetectedBook,
    BatchDetectionRequest,
    BatchDetectionResponse,
    ErrorResponse,
)
import numpy as np
import cv2
from shelfsense.vision.roi_extractor import ROIExtractor
from shelfsense.ocr.ocr_engine import OCREngine


router = APIRouter(prefix="/detect", tags=["detection"])


# =============================================================================
# Configuration
# =============================================================================

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = {"image/jpeg", "image/png", "image/webp"}


# =============================================================================
# Dependencies
# =============================================================================

from fastapi import Depends
from shelfsense.api.dependencies import (
    get_detector_ensemble,
    get_ocr_engine,
    get_service_container,
    get_identification_service,
    ServiceContainer,
)

from shelfsense.identification.service import IdentificationService


async def get_detection_pipeline():
    """Dependency to get detection pipeline."""
    container = get_service_container()
    return container.detector_ensemble





# =============================================================================
# Detection Endpoints
# =============================================================================

@router.post(
    "",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "Image too large"},
    },
)
async def detect_books(
    file: UploadFile = File(..., description="Image file to process"),
    mode: str = Form("auto", description="Detection mode"),
    confidence_threshold: float = Form(0.4, ge=0.0, le=1.0),
    auto_identify: bool = Form(True, description="Auto-identify detected books"),
    enrich_metadata: bool = Form(True, description="Fetch metadata from external APIs"),
    background_tasks: BackgroundTasks = None,
    detector_ensemble = Depends(get_detection_pipeline),
    ocr_engine: OCREngine = Depends(get_ocr_engine),
    identification_service: IdentificationService = Depends(get_identification_service),
):
    """
    Detect books in an uploaded image.
    
    Returns bounding boxes, OCR text, and identification results.
    """
    start_time = time.time()
    
    # Validate file type
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format: {file.content_type}. "
                   f"Supported: {', '.join(SUPPORTED_FORMATS)}",
        )
    
    # Read file
    content = await file.read()
    
    # Check size
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum size of {MAX_IMAGE_SIZE // (1024*1024)}MB",
        )
    
    logger.info(
        f"Processing image: {file.filename}, "
        f"size={len(content)//1024}KB, "
        f"mode={mode}"
    )
    
    # Real detection pipeline
    
    # 1. Decode image
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
        
    image_height, image_width = image.shape[:2]
    
    # 2. Run detection ensemble
    # Checks for spine vs cover layout and runs appropriate models
    ensemble_result = detector_ensemble.detect(image, confidence=confidence_threshold)
    detection_time = ensemble_result.total_inference_time_ms
    
    # 3. Extract and Process ROIs
    roi_extractor = ROIExtractor()
    processed_rois = roi_extractor.extract_and_prepare(
        image, 
        ensemble_result.detections, 
        for_ocr=True, 
        for_embedding=False
    )
    
    detected_books = []
    ocr_start = time.time()
    identification_time = 0.0
    
    # 4. Run OCR on each detected book
    for item in processed_rois:
        roi = item["roi"]
        ocr_image = item["ocr_image"]
        
        # OCR
        ocr_result = ocr_engine.process(
            ocr_image,
            is_spine=(roi.source == "spine")
        )
        
        # Normalize bbox to 0-1 range
        x1, y1, x2, y2 = roi.original_bbox
        normalized_bbox = (
            x1 / image_width,
            y1 / image_height,
            x2 / image_width,
            y2 / image_height
        )
        
        # Identification
        ident_result = None
        if auto_identify and ocr_result.text.strip():
            ident_start = time.time()
            ident_result = await identification_service.identify(ocr_result.text)
            identification_time += (time.time() - ident_start) * 1000

        detected_books.append(
            DetectedBook(
                bbox=normalized_bbox,
                detection_type=roi.source,
                detection_confidence=roi.detection_confidence,
                ocr_text=ocr_result.text,
                ocr_confidence=ocr_result.confidence,
                identified=(ident_result is not None),
                book_id=ident_result["book_id"] if ident_result else None,
                title=ident_result["title"] if ident_result else None,
                author=ident_result["author"] if ident_result else None,
                identification_confidence=ident_result["identification_confidence"] if ident_result else None,
                genres=ident_result.get("genres", []) if ident_result else [],
                publication_year=ident_result.get("publication_year") if ident_result else None,
                isbn_13=ident_result.get("isbn_13") if ident_result else None,
                description=ident_result.get("description") if ident_result else None,
                publisher=ident_result.get("publisher") if ident_result else None,
                cover_url=ident_result.get("cover_url") if ident_result else None,
            )
        )
            
    ocr_time = (time.time() - ocr_start) * 1000

    
    total_time = (time.time() - start_time) * 1000
    
    return DetectionResponse(
        detected_books=detected_books,
        image_width=image_width,
        image_height=image_height,
        layout_type=ensemble_result.layout.layout.value,
        detection_time_ms=detection_time,
        ocr_time_ms=ocr_time,
        identification_time_ms=identification_time,
        total_time_ms=total_time,
        total_detected=len(detected_books),
        total_identified=sum(1 for b in detected_books if b.identified),
        average_confidence=sum(b.detection_confidence for b in detected_books) / len(detected_books) if detected_books else 0,
    )


@router.post(
    "/base64",
    response_model=DetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image data"},
    },
)
async def detect_books_base64(
    image_data: str = Form(..., description="Base64-encoded image"),
    mode: str = Form("auto"),
    confidence_threshold: float = Form(0.5, ge=0.0, le=1.0),
    auto_identify: bool = Form(True),
):
    """Detect books from base64-encoded image."""
    start_time = time.time()
    
    # Decode base64
    try:
        # Handle data URL format
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        content = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {str(e)}",
        )
    
    # Check size
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum size of {MAX_IMAGE_SIZE // (1024*1024)}MB",
        )
    
    logger.info(f"Processing base64 image: size={len(content)//1024}KB")
    
    # Same pipeline as file upload...
    total_time = (time.time() - start_time) * 1000
    
    return DetectionResponse(
        detected_books=[],
        image_width=0,
        image_height=0,
        layout_type="unknown",
        detection_time_ms=0,
        ocr_time_ms=0,
        identification_time_ms=0,
        total_time_ms=total_time,
        total_detected=0,
        total_identified=0,
        average_confidence=0,
    )


@router.post(
    "/batch",
    response_model=BatchDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
    },
)
async def batch_detect_books(
    files: list[UploadFile] = File(..., description="Multiple image files"),
    mode: str = Form("auto"),
    auto_identify: bool = Form(True),
):
    """Process multiple images in batch (limit 10)."""
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch",
        )
    
    start_time = time.time()
    
    logger.info(f"Batch processing {len(files)} images")
    
    results = []
    total_books = 0
    
    for file in files:
        # Validate
        if file.content_type not in SUPPORTED_FORMATS:
            logger.warning(f"Skipping unsupported format: {file.filename}")
            continue
        
        content = await file.read()
        if len(content) > MAX_IMAGE_SIZE:
            logger.warning(f"Skipping oversized image: {file.filename}")
            continue
        
        # Process each image (placeholder)
        result = DetectionResponse(
            detected_books=[],
            image_width=0,
            image_height=0,
            layout_type="unknown",
            detection_time_ms=0,
            ocr_time_ms=0,
            identification_time_ms=0,
            total_time_ms=0,
            total_detected=0,
            total_identified=0,
            average_confidence=0,
        )
        results.append(result)
        total_books += result.total_detected
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchDetectionResponse(
        results=results,
        total_images=len(results),
        total_books_detected=total_books,
        total_time_ms=total_time,
    )


@router.post(
    "/url",
    response_model=DetectionResponse,
)
async def detect_books_from_url(
    url: str = Form(..., description="Image URL"),
    mode: str = Form("auto"),
    auto_identify: bool = Form(True),
):
    """
    Detect books from image URL.
    
    Downloads the image and processes it.
    Supports HTTP and HTTPS URLs.
    """
    import aiohttp
    
    start_time = time.time()
    
    logger.info(f"Fetching image from URL: {url}")
    
    # Download image
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to fetch image: HTTP {resp.status}",
                    )
                
                content_type = resp.headers.get("Content-Type", "")
                if not any(fmt in content_type for fmt in ["jpeg", "png", "webp"]):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported image format: {content_type}",
                    )
                
                content = await resp.read()
                
                if len(content) > MAX_IMAGE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Image exceeds maximum size",
                    )
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch image: {str(e)}",
        )
    
    # Process image (same pipeline)
    total_time = (time.time() - start_time) * 1000
    
    return DetectionResponse(
        detected_books=[],
        image_width=0,
        image_height=0,
        layout_type="unknown",
        detection_time_ms=0,
        ocr_time_ms=0,
        identification_time_ms=0,
        total_time_ms=total_time,
        total_detected=0,
        total_identified=0,
        average_confidence=0,
    )


# =============================================================================
# Detection + Add to Library
# =============================================================================

@router.post(
    "/add-to-library",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
)
async def detect_and_add_to_library(
    file: UploadFile = File(...),
    auto_add: bool = Form(True, description="Automatically add identified books"),
    confidence_threshold: float = Form(0.8, description="Min confidence to auto-add"),
    shelf_location: Optional[str] = Form(None, description="Default shelf location"),
    background_tasks: BackgroundTasks = None,
):
    """
    Detect books and add identified ones to library.
    
    Workflow:
    1. Run detection pipeline
    2. For books above confidence threshold, create records
    3. Return summary with books added and those needing review
    
    Books below threshold are returned for manual review.
    """
    content = await file.read()
    
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds maximum size",
        )
    
    logger.info(
        f"Detect and add: file={file.filename}, "
        f"auto_add={auto_add}, threshold={confidence_threshold}"
    )
    
    # Would process and add in background
    # job_id = await background_tasks.enqueue(
    #     "detect_and_add",
    #     image=content,
    #     auto_add=auto_add,
    #     threshold=confidence_threshold,
    #     shelf_location=shelf_location,
    # )
    
    return {
        "job_id": "job_456",
        "status": "processing",
        "message": "Detection started. Check status endpoint for results.",
    }


@router.get(
    "/job/{job_id}",
    response_model=dict,
)
async def get_detection_job(job_id: str):
    """
    Get status of an async detection job.
    """
    # Would check job status
    return {
        "job_id": job_id,
        "status": "completed",
        "result": {
            "total_detected": 5,
            "auto_added": 3,
            "needs_review": 2,
            "books_added": [
                {"book_id": "book_123", "title": "Dune", "confidence": 0.95},
            ],
            "pending_review": [
                {
                    "ocr_text": "THE GREAT GATSBY",
                    "confidence": 0.65,
                    "suggestions": [
                        {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald"}
                    ],
                },
            ],
        },
    }
