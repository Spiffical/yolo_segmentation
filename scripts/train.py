#!/usr/bin/env python3
"""
Train YOLOv11 segmentation model on MBARI underwater images.

This script provides a clean CLI interface for training with sensible defaults
for both local development and cluster deployment.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo11n-seg.pt',
        help='Base model (yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    # Data arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/yolo_dataset/dataset.yaml',
        help='Path to dataset YAML configuration'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cuda, cpu, 0, 1, etc.)'
    )
    
    # Output arguments
    parser.add_argument(
        '--project',
        type=str,
        default='runs/segment',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )
    
    # Optimization arguments
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate (lr0 * lrf)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        choices=['auto', 'SGD', 'Adam', 'AdamW'],
        help='Optimizer'
    )
    
    # Augmentation arguments
    parser.add_argument(
        '--augment',
        action='store_true',
        default=True,
        help='Enable data augmentation'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    
    # Logging arguments
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        default=True,
        help='Enable TensorBoard logging'
    )
    
    # Misc arguments
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (0 to disable)'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Handle augmentation flag
    augment = args.augment and not args.no_augment
    
    # Auto-detect device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = Path(args.model).stem.replace('.', '_')
        args.name = f'{model_name}_{timestamp}'
    
    print("="*60)
    print("YOLOv11 Segmentation Training")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Output: {args.project}/{args.name}")
    print("="*60)
    
    # Verify dataset exists
    if not os.path.exists(args.data):
        print(f"\nError: Dataset config not found: {args.data}")
        print("Run the conversion script first:")
        print("  python scripts/convert_coco_to_yolo.py")
        sys.exit(1)
    
    # Load model
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\nLoading base model: {args.model}")
        model = YOLO(args.model)
    
    # Setup logging
    loggers = []
    if args.tensorboard:
        loggers.append('tensorboard')
    if args.wandb:
        loggers.append('wandb')
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        verbose=args.verbose,
        augment=augment,
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {args.project}/{args.name}/weights/best.pt")
    print(f"Last model saved to: {args.project}/{args.name}/weights/last.pt")
    
    return results


if __name__ == '__main__':
    main()
