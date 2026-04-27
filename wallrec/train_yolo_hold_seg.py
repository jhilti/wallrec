import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_size(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def resolve_image_path(annotation_path, payload):
    image = payload.get("image")
    if not image:
        raise ValueError(f"{annotation_path} does not contain an image path")

    image_path = Path(image)
    if image_path.is_absolute() and image_path.exists():
        return image_path

    candidates = [
        annotation_path.parent / image_path,
        annotation_path.parent.parent / "images" / image_path.name,
        Path.cwd() / image_path,
        Path.cwd() / "data" / "images" / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find image {image!r} for {annotation_path}")


def simplify_contour(points, epsilon_frac):
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    if epsilon_frac <= 0:
        simplified = contour
    else:
        perimeter = cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon_frac * perimeter, True)
    return simplified.reshape(-1, 2)


def yolo_seg_line(contour, width, height, epsilon_frac):
    points = []
    for point in contour:
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        else:
            x, y = point[:2]
        if x is None or y is None:
            continue
        points.append((float(x), float(y)))

    if len(points) < 3:
        return None

    simplified = simplify_contour(
        np.asarray(points, dtype=np.float32),
        epsilon_frac,
    )
    if len(simplified) < 3:
        return None

    values = ["0"]
    for x, y in simplified:
        nx = min(max(float(x) / width, 0.0), 1.0)
        ny = min(max(float(y) / height, 0.0), 1.0)
        values.extend([f"{nx:.6f}", f"{ny:.6f}"])
    return " ".join(values)


def split_annotations(items, val_ratio, seed):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, shuffled

    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def reset_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_dataset_yaml(dataset_dir):
    yaml_path = dataset_dir / "holds.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: hold",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset(args):
    annotations = [Path(path) for path in args.annotations]
    if not annotations:
        raise ValueError("Pass at least one annotation JSON with --annotations.")

    dataset_dir = Path(args.dataset_dir)
    for subdir in (
        dataset_dir / "images/train",
        dataset_dir / "images/val",
        dataset_dir / "labels/train",
        dataset_dir / "labels/val",
    ):
        reset_dir(subdir)

    train_items, val_items = split_annotations(annotations, args.val_ratio, args.seed)
    splits = {"train": train_items, "val": val_items}
    written = {"train": 0, "val": 0}
    labels = {"train": 0, "val": 0}

    for split, paths in splits.items():
        for annotation_path in paths:
            payload = read_json(annotation_path)
            source_image = resolve_image_path(annotation_path, payload)
            width, height = image_size(source_image)

            image_name = source_image.name
            if (dataset_dir / f"images/{split}" / image_name).exists():
                image_name = f"{annotation_path.stem}_{source_image.name}"

            target_image = dataset_dir / f"images/{split}" / image_name
            target_label = dataset_dir / f"labels/{split}" / f"{Path(image_name).stem}.txt"
            copy_image(source_image, target_image)

            lines = []
            for detection in payload.get("detections", []):
                line = yolo_seg_line(
                    detection.get("contour", []),
                    width,
                    height,
                    args.epsilon_frac,
                )
                if line:
                    lines.append(line)

            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written[split] += 1
            labels[split] += len(lines)

    yaml_path = write_dataset_yaml(dataset_dir)
    print(f"Dataset: {dataset_dir}")
    print(f"Config: {yaml_path}")
    print(f"Train images: {written['train']} ({labels['train']} masks)")
    print(f"Val images: {written['val']} ({labels['val']} masks)")
    return yaml_path


def train_model(args, data_yaml):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics first: .venv/bin/pip install ultralytics") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=None if args.device == "auto" else args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        task="segment",
        exist_ok=args.exist_ok,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Best checkpoint: {best}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO segmentation dataset from hold JSONs and train a hold detector."
    )
    parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="Detected-holds JSON files to use as segmentation annotations.",
    )
    parser.add_argument("--dataset-dir", default="datasets/holds", help="Output YOLO dataset directory.")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="YOLO segmentation model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, 0, 0,1, etc.")
    parser.add_argument("--project", default="runs/segment")
    parser.add_argument("--name", default="hold-yolo")
    parser.add_argument(
        "--no-exist-ok",
        action="store_false",
        dest="exist_ok",
        help="Let Ultralytics create an incremented run directory if the name exists.",
    )
    parser.set_defaults(exist_ok=True)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--epsilon-frac",
        type=float,
        default=0.002,
        help="Polygon simplification as a fraction of contour perimeter. Use 0 to keep all points.",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Only write the YOLO dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_yaml = prepare_dataset(args)
    if not args.prepare_only:
        train_model(args, data_yaml)


if __name__ == "__main__":
    main()
