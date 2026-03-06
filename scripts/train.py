#!/usr/bin/env python3
"""
Train Ultralytics YOLO detection or segmentation models on MBARI imagery.

This script provides a clean CLI interface for both local development and
cluster deployment.
"""

import os
import shutil
import sys
import argparse
from pathlib import Path
from datetime import datetime

if "YOLO_CONFIG_DIR" not in os.environ:
    local_config_dir = (Path(__file__).resolve().parent.parent / "runs" / "ultralytics").resolve()
    local_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(local_config_dir)

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
        description="Train a YOLO detection or segmentation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo11n-seg.pt',
        help='Base model or checkpoint path (e.g. yolo11m.pt, yolo11m-seg.pt, /path/to/best.pt)'
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
        default=None,
        help='Project directory for saving results (default: runs/<task>)'
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
    
    # Validation-time augmentation arguments
    parser.add_argument(
        '--augment',
        action='store_true',
        default=False,
        help='Enable test-time augmentation (TTA) during validation/prediction'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable validation/prediction TTA (default)'
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
        '--snapshot-best-confusion',
        dest='snapshot_best_confusion',
        action='store_true',
        default=True,
        help='Generate confusion matrix snapshots only when best.pt is updated'
    )
    parser.add_argument(
        '--no-snapshot-best-confusion',
        dest='snapshot_best_confusion',
        action='store_false',
        help='Disable best.pt confusion matrix snapshots'
    )
    parser.add_argument(
        '--best-confusion-split',
        type=str,
        default='val',
        help='Dataset split to use for best.pt confusion matrix snapshots'
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
    
    # Handle validation-time TTA flag.
    # Training augmentation is controlled by Ultralytics hypers, not this flag.
    val_augment = args.augment and not args.no_augment
    
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
    
    # Verify dataset exists
    if not os.path.exists(args.data):
        print(f"\nError: Dataset config not found: {args.data}")
        print("Run the conversion script first:")
        print("  python scripts/convert_coco_to_yolo.py")
        sys.exit(1)
    
    # Load model
    resume_ckpt = None
    if args.resume:
        resume_ckpt = Path(args.resume).expanduser().resolve()
        if not resume_ckpt.exists():
            print(f"\nError: Resume checkpoint not found: {resume_ckpt}")
            print("Pass a valid path to weights/last.pt for full-state resume.")
            sys.exit(1)
        print(f"\nResuming from checkpoint: {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
    else:
        print(f"\nLoading base model: {args.model}")
        model = YOLO(args.model)

    task_name = getattr(model, 'task', 'train')
    if args.project is None:
        args.project = f"runs/{task_name}"

    print("="*60)
    print(f"YOLO {task_name.capitalize()} Training")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Task: {task_name}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Validation TTA: {val_augment}")
    print(f"Output: {args.project}/{args.name}")
    print("="*60)

    if args.snapshot_best_confusion:
        best_state = {'signature': None}

        def _force_epoch_plots(trainer):
            """Ensure confusion matrix plots are generated during each epoch validation."""
            if not hasattr(trainer, 'validator'):
                return
            trainer.validator.args.split = args.best_confusion_split
            trainer.validator.args.plots = True
            trainer.stopper.possible_stop = True

        def snapshot_best_confusion(trainer):
            best_path = Path(trainer.best)
            if not best_path.exists():
                return

            best_stat = best_path.stat()
            signature = (best_stat.st_mtime_ns, best_stat.st_size)
            if signature == best_state['signature']:
                return
            best_state['signature'] = signature

            epoch_num = int(getattr(trainer, 'epoch', -1)) + 1
            print(
                f"[best-snapshot] best.pt updated at epoch {epoch_num}; "
                f"capturing confusion matrices for split='{args.best_confusion_split}'."
            )

            snapshot_dir = Path(trainer.save_dir) / 'best_snapshots' / f'epoch_{epoch_num:04d}'
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            copied = []
            for filename in ('confusion_matrix.png', 'confusion_matrix_normalized.png'):
                src = Path(trainer.save_dir) / filename
                if src.exists():
                    dst = snapshot_dir / filename
                    shutil.copy2(src, dst)
                    copied.append(str(dst))

            pointer_path = snapshot_dir / 'best_checkpoint.txt'
            pointer_path.write_text(
                f"path: {best_path}\n"
                f"epoch: {epoch_num}\n"
                f"mtime_ns: {best_stat.st_mtime_ns}\n"
                f"size_bytes: {best_stat.st_size}\n",
                encoding='utf-8'
            )

            if copied:
                print("[best-snapshot] saved:")
                for path in copied:
                    print(f"  - {path}")
                print(f"  - {pointer_path}")
            else:
                print("[best-snapshot] WARNING: confusion matrix files were not found after epoch validation.")

        model.add_callback('on_train_epoch_end', _force_epoch_plots)
        model.add_callback('on_model_save', snapshot_best_confusion)
        print(f"Best-checkpoint confusion snapshots: enabled (split={args.best_confusion_split})")
    else:
        print("Best-checkpoint confusion snapshots: disabled")
    
    # Setup logging
    # Configure explicit W&B run metadata if requested or if WANDB env vars are set.
    # We keep Ultralytics' W&B callback enabled so validation curves/plots are logged.
    use_wandb = args.wandb or os.environ.get('WANDB_PROJECT') is not None
    wandb = None
    settings = None
    
    if use_wandb:
        try:
            import wandb
            from ultralytics import settings

            settings.update({'wandb': True})
            
            # Get project/run name from env or defaults
            wandb_project = os.environ.get('WANDB_PROJECT', 'yolo-segmentation')
            wandb_name = os.environ.get('WANDB_NAME', args.name)
            wandb_group = os.environ.get('WANDB_GROUP')
            wandb_tags_raw = os.environ.get('WANDB_TAGS', '')
            wandb_tags = [t.strip() for t in wandb_tags_raw.split(',') if t.strip()]
            
            print(f"W&B logging enabled: project={wandb_project}, run={wandb_name}")
            if wandb_group:
                print(f"W&B group: {wandb_group}")
            if wandb_tags:
                print(f"W&B tags: {wandb_tags}")
            
            # Pre-initialize run so Ultralytics callbacks reuse this metadata
            # instead of inferring project from output filesystem path.
            if wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    group=wandb_group,
                    tags=wandb_tags,
                    config={
                        'model': args.model,
                        'epochs': args.epochs,
                        'batch': args.batch,
                        'imgsz': args.imgsz,
                        'lr0': args.lr0,
                        'optimizer': args.optimizer,
                    }
                )
        except ImportError:
            print("W&B requested but wandb not installed. Skipping.")
            use_wandb = False
            try:
                from ultralytics import settings
                settings.update({'wandb': False})
            except Exception:
                pass
        except Exception as e:
            print(f"W&B initialization failed: {e}")
            use_wandb = False
            try:
                if settings is not None:
                    settings.update({'wandb': False})
            except Exception:
                pass
    else:
        try:
            from ultralytics import settings
            settings.update({'wandb': False})
        except Exception:
            pass
    
    # Train
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=(resume_ckpt is None),
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        save_period=args.save_period,
        seed=args.seed,
        verbose=args.verbose,
        augment=val_augment,
        val=True,
        plots=True,
    )
    if resume_ckpt is not None:
        # True Ultralytics resume: restores optimizer/scheduler/epoch state.
        # Use last.pt (not best.pt) for this.
        train_kwargs['resume'] = True

    results = model.train(**train_kwargs)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    save_dir = Path(args.project) / args.name
    try:
        if getattr(model, 'trainer', None) is not None and getattr(model.trainer, 'save_dir', None) is not None:
            save_dir = Path(model.trainer.save_dir)
    except Exception:
        pass
    print(f"Best model saved to: {save_dir}/weights/best.pt")
    print(f"Last model saved to: {save_dir}/weights/last.pt")
    
    return results


if __name__ == '__main__':
    main()
