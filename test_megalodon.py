
import os
import argparse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import glob

def setup_model(model_name="mbari-megalodon-yolov8x.pt"):
    print(f"Checking for model: {model_name}...")
    try:
        # Try to download from HF
        model_path = hf_hub_download(repo_id="FathomNet/megalodon", filename=model_name)
        print(f"Model found at: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def run_inference(source, model_path, output_dir="runs/detect"):
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running inference on {source}...")
    
    # Run inference
    # save=True will save the annotated images/videos to runs/detect/exp...
    try:
        results = model.predict(source, save=True, project=output_dir, name="megalodon_test", exist_ok=True)
        print(f"Inference complete. Results saved to {output_dir}/megalodon_test")
        return results
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test FathomNet Megalodon Model")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video file, or directory")
    args = parser.parse_args()
    
    model_path = setup_model()
    if not model_path:
        print("Could not set up model. Exiting.")
        return

    run_inference(args.source, model_path)

if __name__ == "__main__":
    main()
