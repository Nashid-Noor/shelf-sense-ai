"""
Debug OCR Quality Script

Analyzes detection and OCR performance on a specific image.
"""
import os
import sys
import cv2
import asyncio
from dotenv import load_dotenv
from loguru import logger
from dataclasses import asdict

# Ensure current dir is in path
sys.path.append(os.getcwd())

# Load env
load_dotenv()

# Filter logs
logger.remove()
logger.add(sys.stderr, level="INFO")

from shelfsense.api.dependencies import get_settings, init_services
from shelfsense.vision.roi_extractor import ROIExtractor

IMAGE_PATH = "/Users/noormohammed/.gemini/antigravity/brain/de6bddc5-7fe4-46b6-9a73-3eae80f8e453/uploaded_media_1_1769963080038.jpg"

def main():
    print("="*50)
    print("OCR DEBUG ANALYSIS")
    print("="*50)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        return

    # Init services
    print("Initializing services...")
    settings = get_settings()
    container = init_services(settings)
    detector = container.detector_ensemble
    ocr_engine = container.ocr_engine
    roi_extractor = ROIExtractor()
    
    # Load image
    print(f"Loading image: {os.path.basename(IMAGE_PATH)}")
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print("❌ Failed to load image")
        return
    print(f"Image shape: {image.shape}")
    
    # 1. Detect
    print("\n1. Running Detection...")
    det_result = detector.detect(image)
    print(f"Found {len(det_result.detections)} detections")
    
    if not det_result.detections:
        print("No books detected.")
        return

    # 2. Extract ROIs
    print("\n2. Extracting ROIs...")
    processed_items = roi_extractor.extract_and_prepare(
        image, 
        det_result.detections,
        for_ocr=True,
        for_embedding=False
    )
    
    # Sort by X coordinate/shelf order
    processed_items.sort(key=lambda x: x['roi'].original_bbox[0])

    print(f"\n3. Running OCR on {len(processed_items)} items...")
    print("-" * 60)
    print(f"{'ID':<4} | {'Source':<8} | {'Conf':<6} | {'OCR Text':<30} | {'OCR Conf':<8}")
    print("-" * 60)
    
    for i, item in enumerate(processed_items):
        roi = item['roi']
        ocr_image = item['ocr_image']
        
        # Run OCR
        ocr_result = ocr_engine.process(
            ocr_image,
            is_spine=(roi.source == "spine")
        )
        
        text = ocr_result.text.strip()
        text_display = (text[:27] + '...') if len(text) > 27 else text
        if not text_display:
            text_display = "[NO TEXT]"
            
        print(f"{i+1:<4} | {roi.source:<8} | {roi.detection_confidence:.2f}   | {text_display:<30} | {ocr_result.confidence:.2f}")
        
        # Debug low confidence items
        if ocr_result.confidence < 0.5 or not text:
            logger.debug(f"Input shape: {ocr_image.shape}, Engine: {ocr_result.engine_used}")

        # Draw on image
        x1, y1, x2, y2 = roi.original_bbox
        # Color: Green for high confidence, Red for low/empty
        color = (0, 255, 0) if text and ocr_result.confidence > 0.6 else (0, 0, 255)
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        label = f"{text[:15]} ({ocr_result.confidence:.2f})"
        cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = "debug_result.jpg"
    cv2.imwrite(output_path, image)
    print("-" * 60)
    print(f"Saved visualization to: {os.path.abspath(output_path)}")
    print("Done.")

if __name__ == "__main__":
    main()
