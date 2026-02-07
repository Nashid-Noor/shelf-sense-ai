import argparse
import cv2
import os
import sys
from pathlib import Path

# Add project root to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shelfsense.vision.spine_detector import SpineDetector
from shelfsense.api.dependencies import get_settings

def test_on_image(image_path, output_path="output.jpg"):
    """
    Run detection on a single image and save the result.
    """
    settings = get_settings()
    print(f"Loading model from: {settings.yolo_model_path}")
    
    try:
        detector = SpineDetector(model_path=settings.yolo_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    # Run detection
    result = detector.detect(image)
    print(f"Found {len(result.detections)} spines.")

    # Visualize results
    # SpineDetector.visualize returns a new image with boxes drawn
    annotated_image = detector.visualize(image, result)

    # Save output
    cv2.imwrite(output_path, annotated_image)
    print(f"Saved annotated image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test spine detection on an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", default="output.jpg", help="Path to save the output image")
    
    args = parser.parse_args()
    
    test_on_image(args.image_path, args.output)
