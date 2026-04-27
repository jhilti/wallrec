# wallrec

Hold detection experiments for climbing wall photos.

## Project Layout

```text
wallrec/            Python modules and runnable workflows
data/images/        Example wall images
data/annotations/   Reviewed or reference hold annotations
models/             Local model checkpoints (ignored by git)
outputs/            New detector outputs (ignored by git)
reports/plots/      Debug and comparison plots
datasets/           Generated YOLO datasets (ignored by git)
runs/               Ultralytics training runs (ignored by git)
```

## Detect Holds

```bash
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend classic --target-count 92
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend sam-vit-b --checkpoint models/sam_vit_b_01ec64.pth --target-count 92
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend mobile-sam --checkpoint models/mobile_sam.pt --target-count 92
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend efficient-sam --checkpoint models/efficient_sam_vitt.pt --target-count 92
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend fastsam --yolo-model models/FastSAM-s.pt --target-count 92
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend yolo-seg --yolo-model runs/segment/hold-yolo/weights/best.pt --target-count 92
```

New detection JSONs are written to `outputs/annotations/` unless `--save-holds` is provided.

## Train YOLO Hold Segmentation

Create a YOLO segmentation dataset from reviewed hold JSON files:

```bash
.venv/bin/python -m wallrec.train_yolo_hold_seg \
  --annotations data/annotations/IMG_1637_mobile_sam_detected_holds.json data/annotations/IMG_1505_detected_holds.json \
  --dataset-dir datasets/holds \
  --prepare-only
```

Train a small YOLO segmentation model:

```bash
.venv/bin/python -m wallrec.train_yolo_hold_seg \
  --annotations data/annotations/IMG_1637_mobile_sam_detected_holds.json data/annotations/IMG_1505_detected_holds.json \
  --dataset-dir datasets/holds \
  --model yolo11n-seg.pt \
  --epochs 100 \
  --imgsz 1024 \
  --batch 4 \
  --device auto
```

Use the trained checkpoint:

```bash
.venv/bin/python -m wallrec.hold_detection_workflow data/images/IMG_1637.jpeg --backend yolo-seg --yolo-model runs/segment/hold-yolo/weights/best.pt --target-count 92
```
