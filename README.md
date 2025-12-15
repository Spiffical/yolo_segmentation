
# YOLO Segmentation & FathomNet Tools

This repository contains tools for working with the FathomNet dataset and the MBARI Megalodon object detection model (YOLOv8).

## Setup

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the environment:**
    *   Linux/macOS:
        ```bash
        source .venv/bin/activate
        ```
    *   Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Download FathomNet Images
Downloads images corresponding to a COCO-format annotation file (e.g., `train.json`).

```bash
python download_images.py --json_path data/seg_masks/train.json --save_dir data/images
```
*   `--json_path`: Path to your COCO JSON file.
*   `--save_dir`: Directory where images will be saved.
*   `--workers`: Number of parallel download threads (default: 16).

### 2. Run Megalodon (Object Detection)
Runs the pre-trained FathomNet Megalodon model (YOLOv8x) on images or videos. It automatically downloads the model weights from Hugging Face if not present.

```bash
python test_megalodon.py --source data/images/sample.png
# OR
python test_megalodon.py --source data/videos/my_video.mp4
```
*   `--source`: Path to an image, video, or directory of images.
*   Results will be saved to `runs/detect/`.

## Data Structure
The scripts expect a structure similar to this (though paths are configurable):
```
├── data/
│   ├── images/       # Downloaded images
│   ├── seg_masks/    # COCO JSON annotation files
│   └── videos/       # Input videos for testing
├── runs/             # Inference results
```
