# Coarse Label Plan for MBARI/FathomNet

## Dataset Facts

- Source file profiled: `data/seg_masks/train.json`
- Images: `119,096`
- Annotations: `280,118`
- Declared categories: `1,897`
- Median annotations per class: `6`
- Built-in `supercategory` is not usable here; in practice it is `none`
- The label space is mixed-resolution: species, genus, higher taxa, common names, and non-biological objects all appear together
- The taxonomy is overlapping: there are at least `530` parent/child name pairs like `Bathochordaeus` and `Bathochordaeus mcnutti`

Implication: a flat 1,897-way model is a poor first target. The safer progression is binary detection, then coarse detection, then finer label sets.

## Proposed Label Plans

### Plan A: `binary_objectness`

Use every annotation as a single `object` class.

- Why: best first baseline for recall and easiest way to validate the bbox pipeline
- Coverage: effectively all annotations
- Primary checkpoint: `FathomNet/megalodon`
- Secondary checkpoints: `FathomNet/MBARI-315k-yolov8`, `yolo26l.pt`, `yolo11l.pt`

### Plan B: `coarse_v1_bio8`

Keep only visually coherent biological groups and drop unresolved tail labels for the first pass.

| coarse label | approximate annotations | examples |
|---|---:|---|
| `echinoderm` | 57,815 | `Strongylocentrotus fragilis`, `Ophiuroidea`, `Peniagone`, `Asteroidea` |
| `coral_anemone` | 53,027 | `Alcyonacea`, `Actiniaria`, `Paragorgia arborea`, `Pennatulacea` |
| `gelatinous` | 25,389 | `Bathochordaeus`, `Solmissus`, `medusa`, `Pyrosoma`, `Nanomia bijuga` |
| `sponge` | 24,958 | `Porifera`, `sponge`, `Hexactinellida`, `Heterochone calyx` |
| `fish` | 22,100 | `Anoplopoma`, `Sebastolobus`, `Merluccius`, `Coryphaenoides` |
| `crustacean` | 16,121 | `Chionoecetes tanneri`, `Munidopsis`, `Caridea`, `shrimp` |
| `other_invertebrate` | 9,450 | `Sabellidae`, `Serpulidae`, `Gastropoda`, `Vesicomyidae` |
| `cephalopod` | 3,876 | `Octopus`, `Vampyroteuthis`, `Chiroteuthis calyx` |

- Approximate coverage: `212,736` annotations, about `75.9%` of the dataset
- Dropped for v1: ambiguous or very long-tail labels that do not map cleanly into a visually consistent bucket
- Primary checkpoint: `FathomNet/MBARI-315k-yolov8`
- Secondary checkpoints: `yolo26l.pt`, `yolo11l.pt`

### Plan C: `coarse_v1_bio8_plus_gear`

Same as `coarse_v1_bio8`, plus one nuisance class for man-made objects.

| coarse label | approximate annotations | examples |
|---|---:|---|
| `gear_debris` | 1,948 | `equipment`, `manipulator`, `trash`, `cable`, `bottle`, `can` |

- Total coverage with gear: `214,684` annotations, about `76.6%` of the dataset
- Use this only if false positives on ROV hardware, tools, and debris matter to the downstream task
- Primary checkpoint: `FathomNet/MBARI-315k-yolov8`
- Optional auxiliary checkpoint: `FathomNet/trash-detector`

### Plan D: `top_50_head_labels`

Train the top `50` native labels directly without coarse remapping.

- Coverage: about `60.6%` of all annotations
- The 50th class still has `1,173` annotations
- Good for testing whether native head labels outperform coarse groups
- Primary checkpoint: `FathomNet/MBARI-315k-yolov8`

## Recommended Fine-Tuning Tracks

### Track 1: Binary Detection

Train first.

- Goal: maximize object recall and validate conversion, training, and evaluation
- Model shortlist:
  - `FathomNet/megalodon`
  - `FathomNet/MBARI-315k-yolov8`
  - `yolo26l.pt`
  - `yolo11l.pt`

Expected outcome: this should be the strongest early model if the downstream need is simply "find marine things in the frame."

### Track 2: Coarse Detection

Train second.

- Default label plan: `coarse_v1_bio8`
- Add `gear_debris` only if nuisance detections on hardware or trash are a real problem
- Model shortlist:
  - `FathomNet/MBARI-315k-yolov8`
  - `yolo26l.pt`
  - `yolo11l.pt`

Expected outcome: this is the most practical compromise between biological meaning and model stability.

### Track 3: Habitat-Specific Models

Train only after you know whether the deployment is mostly benthic or mostly midwater.

- Midwater-oriented subset:
  - likely strongest starting checkpoint: `FathomNet/2025-MBARI-Midwater-Supercategory-Object-Detector`
  - backup: `FathomNet/MBARI-midwater-supercategory-detector`
- Benthic-oriented subset:
  - practical high-level checkpoint: `FathomNet/vulnerable-marine-ecosystems`
  - broader backup: `FathomNet/MBARI-315k-yolov8`

Expected outcome: specialized models may outperform one global coarse detector if your images come from a narrower habitat regime.

### Track 4: Segmentation Follow-Up

Run only after detection baselines are stable.

- Label plan: reuse `binary_objectness` or `coarse_v1_bio8`
- Model shortlist:
  - `yolo26l-seg.pt`
  - `yolo26x-seg.pt`
  - `yolo11x-seg.pt` if version compatibility is easier

Expected outcome: segmentation may improve localization quality, but the mask release is newer and likely noisier than the original bbox labels.

## Recommended Order

1. Fine-tune `FathomNet/megalodon` on `binary_objectness`.
2. Fine-tune `FathomNet/MBARI-315k-yolov8` on `coarse_v1_bio8`.
3. Compare against a generic baseline on the same labels with `yolo26l.pt`.
4. If the deployment is clearly midwater or clearly benthic, branch into a habitat-specific model.
5. Only after that, decide whether instance segmentation is worth the extra cost.

## Practical Notes for This Repo

- The current repo already supports segmentation labels.
- Detection training is feasible here, but the COCO-to-YOLO conversion path needs a bbox mode.
- The coarse plans above are intended to drive that conversion step.
- For the first conversion pass, it is better to map only confident labels and drop ambiguous classes than to force every rare taxon into a noisy bucket.
