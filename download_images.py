
import json
import os
import fathomnet.api.images
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
import argparse

def download_image(image_info, save_dir):
    file_name = image_info['file_name']
    image_uuid = os.path.splitext(file_name)[0]
    save_path = os.path.join(save_dir, file_name)
    
    # Skip if already exists and is valid (size > 0)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return
        
    try:
        # Fathomnet API to get image details, which should include the URL
        img_record = fathomnet.api.images.find_by_uuid(image_uuid)
        if img_record and img_record.url:
            response = requests.get(img_record.url, stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                # print(f"Failed to download {image_uuid}: Status {response.status_code}")
                pass
        else:
            # print(f"No record found for {image_uuid}")
            pass
            
    except Exception as e:
        # print(f"Error downloading {image_uuid}: {e}")
        pass

def main():
    parser = argparse.ArgumentParser(description="Download images from FathomNet based on COCO JSON")
    parser.add_argument('--json_path', type=str, default='data/seg_masks/train.json', help='Path to COCO JSON file')
    parser.add_argument('--save_dir', type=str, default='data/images', help='Directory to save images')
    parser.add_argument('--workers', type=int, default=16, help='Number of threaded workers')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Loading {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
        
    images = data['images']
    print(f"Found {len(images)} images to download.")
    
    print(f"Downloading to {args.save_dir} with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(tqdm(executor.map(lambda x: download_image(x, args.save_dir), images), total=len(images)))

if __name__ == "__main__":
    main()
