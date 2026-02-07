import os
import subprocess
import sys
import argparse
import yaml

def install_dependencies():
    """Install minimal dependencies for Colab/Training"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "roboflow", "-q"])

def download_dataset(api_key):
    """Download verified public dataset from Roboflow"""
    print("Downloading dataset...")
    from roboflow import Roboflow
    
    rf = Roboflow(api_key=api_key)
    # Using user's personal fork (Verified)
    project = rf.workspace("personal-project-jf247").project("book-spine-detection-2cci9-svccl")
    dataset = project.version(1).download("yolov8")
    
    return dataset.location

def fix_yaml_paths(dataset_path):
    """Ensure data.yaml uses absolute paths for Colab"""
    yaml_path = f"{dataset_path}/data.yaml"
    
    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found. Checking if inside a subdir...")
        # Sometimes it's nested
        for root, dirs, files in os.walk(dataset_path):
            if "data.yaml" in files:
                yaml_path = os.path.join(root, "data.yaml")
                dataset_path = root
                break
    
    print(f"Fixing YAML at: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to be absolute
    data['path'] = os.path.abspath(dataset_path)
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    data['test'] = 'test/images'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    return yaml_path

def train_model(data_yaml, epochs=50):
    """Train YOLOv8n"""
    from ultralytics import YOLO
    
    print(f"Starting training on {data_yaml}...")
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        project="spine_detector",
        name="train_run",
        # Augmentations
        degrees=5.0,
        fliplr=0.5,
        mosaic=1.0,
    )
    print(f"Training complete! Weights saved to spine_detector/train_run/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated with user's key for convenience
    parser.add_argument("--api-key", type=str, default="F3gn8PQb0mw8xFwNOqlm", help="Roboflow API Key")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    try:
        install_dependencies()
        dataset_dir = download_dataset(args.api_key)
        final_yaml = fix_yaml_paths(dataset_dir)
        train_model(final_yaml, args.epochs)
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: Sign up at https://app.roboflow.com to get a free API Key.")
