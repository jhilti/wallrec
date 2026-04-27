import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from wallrec.auto_hold_detector import bbox_iou, centers_too_close, load_image
from wallrec.lightweight_hold_detector_1637 import (
    build_search_hold_score,
    order_corners,
    pick_peak_points,
    polygon_mask,
    propose_detections,
    region_to_detection,
    resize_image,
    select_final_detections as select_banded_detections,
)


BACKENDS = (
    "classic",
    "sam",
    "sam-vit-b",
    "mobile-sam",
    "efficient-sam",
    "fastsam",
    "yolo-seg",
)

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")

DEFAULT_SIZE_REFERENCE = ANNOTATIONS_DIR / "IMG_1637_mobile_sam_detected_holds.json"


def pick_torch_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_torch_numpy_compat(torch):
    original_as_tensor = torch.as_tensor
    dtype_map = {
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.bool_): torch.bool,
    }

    def patched_as_tensor(data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            device = kwargs.pop("device", None)
            dtype = kwargs.pop("dtype", dtype_map.get(data.dtype))
            if data.dtype == np.uint8 and data.flags["C_CONTIGUOUS"]:
                tensor = torch.frombuffer(bytearray(data.tobytes()), dtype=torch.uint8).clone().reshape(data.shape)
                if dtype is not None and dtype != torch.uint8:
                    tensor = tensor.to(dtype=dtype)
            else:
                tensor = torch.tensor(data.tolist(), dtype=dtype)
            if device is not None:
                tensor = tensor.to(device)
            return tensor
        return original_as_tensor(data, *args, **kwargs)

    torch.as_tensor = patched_as_tensor

    def patched_numpy(self):
        return np.asarray(self.detach().cpu().tolist())

    def patched_array(self, dtype=None):
        array = np.asarray(self.detach().cpu().tolist())
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    torch.Tensor.numpy = patched_numpy
    torch.Tensor.__array__ = patched_array


def patch_sam_predictor_set_image(torch, predictor_cls):
    def patched_set_image(self, image, image_format="RGB"):
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.frombuffer(bytearray(input_image.tobytes()), dtype=torch.uint8).clone()
        input_image_torch = input_image_torch.reshape(input_image.shape).to(self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.set_torch_image(input_image_torch, image.shape[:2])

    predictor_cls.set_image = patched_set_image


def collect_corners_interactively(image_rgb):
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(image_rgb)
    ax.set_title("Click the 4 wall corners. Close or press q to cancel.")
    ax.axis("off")

    corners = []
    artists = []

    def redraw():
        while artists:
            artists.pop().remove()
        if corners:
            pts = np.asarray(corners, dtype=np.float32)
            artists.append(ax.scatter(pts[:, 0], pts[:, 1], c="#ff3b30", s=60))
            for idx, (x, y) in enumerate(corners, start=1):
                artists.append(
                    ax.text(
                        x + 8,
                        y - 8,
                        str(idx),
                        color="white",
                        fontsize=10,
                        bbox={"facecolor": "#ff3b30", "edgecolor": "none", "pad": 1.5},
                    )
                )
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        corners.append([float(event.xdata), float(event.ydata)])
        redraw()
        if len(corners) == 4:
            plt.close(fig)

    def onkey(event):
        if event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()

    if len(corners) != 4:
        raise RuntimeError("Expected exactly 4 corner clicks.")
    return order_corners(corners)


def build_context(image_rgb, corners, max_height, size_constraints=None):
    image_small, scale = resize_image(image_rgb, max_height=max_height)
    corners = order_corners(np.asarray(corners, dtype=np.float32))
    corners_small = corners / scale
    search_mask = polygon_mask(image_small.shape, corners_small)
    hold_score = build_search_hold_score(image_small, search_mask)
    return {
        "image_rgb": image_rgb,
        "image_small": image_small,
        "scale": scale,
        "corners": corners,
        "corners_small": corners_small,
        "search_mask": search_mask,
        "hold_score": hold_score,
        "size_constraints": size_constraints,
        "backend_state": {},
    }


def contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def reference_detection_metrics(detection):
    contour = np.asarray(detection["contour"], dtype=np.float32).reshape(-1, 1, 2)
    x0, y0, x1, y1 = detection["bbox"]
    width = float(x1 - x0)
    height = float(y1 - y0)
    return {
        "area": float(detection["area"]),
        "perimeter": float(cv2.arcLength(contour, True)),
        "max_dim": max(width, height),
        "min_dim": min(width, height),
    }


def load_size_constraints(reference_path, min_tolerance, max_linear_scale):
    if not reference_path:
        return None

    path = Path(reference_path)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    detections = payload.get("detections", [])
    if not detections:
        return None

    metrics = [reference_detection_metrics(det) for det in detections]
    values = {key: np.asarray([item[key] for item in metrics], dtype=np.float32) for key in metrics[0]}
    max_area_scale = max_linear_scale * max_linear_scale

    return {
        "source": str(path),
        "count": len(detections),
        "min_tolerance": float(min_tolerance),
        "max_linear_scale": float(max_linear_scale),
        "min_area": float(values["area"].min() * min_tolerance),
        "min_perimeter": float(values["perimeter"].min() * min_tolerance),
        "min_min_dim": float(values["min_dim"].min() * min_tolerance),
        "min_max_dim": float(values["max_dim"].min() * min_tolerance),
        "max_area": float(values["area"].max() * max_area_scale),
        "max_perimeter": float(values["perimeter"].max() * max_linear_scale),
        "max_dim": float(values["max_dim"].max() * max_linear_scale),
        "reference_min": {key: float(value.min()) for key, value in values.items()},
        "reference_max": {key: float(value.max()) for key, value in values.items()},
    }


def detection_size_metrics(detection, scale):
    x0, y0, x1, y1 = detection["bbox"]
    width = float(x1 - x0) * scale
    height = float(y1 - y0) * scale
    contour = np.asarray(detection["contour"], dtype=np.float32)
    return {
        "area": float(detection["area"]) * scale * scale,
        "perimeter": float(cv2.arcLength(contour, True)) * scale,
        "max_dim": max(width, height),
        "min_dim": min(width, height),
    }


def size_rejection_reason(detection, constraints, scale):
    if not constraints:
        return None

    metrics = detection_size_metrics(detection, scale)
    checks = (
        ("area", "min_area", "<"),
        ("perimeter", "min_perimeter", "<"),
        ("min_dim", "min_min_dim", "<"),
        ("max_dim", "min_max_dim", "<"),
        ("area", "max_area", ">"),
        ("perimeter", "max_perimeter", ">"),
        ("max_dim", "max_dim", ">"),
    )
    for metric_key, limit_key, operator in checks:
        value = metrics[metric_key]
        limit = constraints[limit_key]
        epsilon = 1e-3
        if operator == "<" and value < limit - epsilon:
            return f"{metric_key} {value:.1f} below {limit:.1f}"
        if operator == ">" and value > limit + epsilon:
            return f"{metric_key} {value:.1f} above {limit:.1f}"
    return None


def passes_size_constraints(detection, ctx):
    return size_rejection_reason(detection, ctx.get("size_constraints"), ctx["scale"]) is None


def filter_size_constraints(detections, ctx):
    constraints = ctx.get("size_constraints")
    if not constraints:
        return detections
    return [det for det in detections if passes_size_constraints(det, ctx)]


def describe_size_constraints(constraints):
    if not constraints:
        return "disabled"
    return (
        f"{constraints['source']} ({constraints['count']} holds): "
        f"area {constraints['min_area']:.0f}-{constraints['max_area']:.0f}px, "
        f"perimeter {constraints['min_perimeter']:.0f}-{constraints['max_perimeter']:.0f}px, "
        f"max dimension {constraints['min_max_dim']:.0f}-{constraints['max_dim']:.0f}px"
    )


def normalize_detection(det, backend):
    item = dict(det)
    item["bbox"] = tuple(int(v) for v in item["bbox"])
    item["center"] = (float(item["center"][0]), float(item["center"][1]))
    item["area"] = float(item["area"])
    item["confidence"] = float(item.get("confidence", 0.0))
    item["peak_score"] = float(item.get("peak_score", item["confidence"]))
    item["mean_score"] = float(item.get("mean_score", item["confidence"] * 0.75))
    item["backend"] = backend
    item["contour"] = np.asarray(item["contour"], dtype=np.float32).reshape(-1, 1, 2)
    return item


def scale_detection(det, scale):
    contour = np.asarray(det["contour"], dtype=np.float32).copy()
    contour[:, 0, 0] *= scale
    contour[:, 0, 1] *= scale
    x0, y0, x1, y1 = det["bbox"]
    cx, cy = det["center"]
    return {
        "bbox": (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        ),
        "center": (float(cx * scale), float(cy * scale)),
        "area": float(det["area"] * scale * scale),
        "confidence": float(det.get("confidence", 0.0)),
        "backend": det.get("backend", ""),
        "source": det.get("source", ""),
        "contour": contour,
    }


def detections_to_full(detections_small, scale):
    return [scale_detection(det, scale) for det in detections_small]


def point_inside_detection(detection, x, y):
    contour = np.asarray(detection["contour"], dtype=np.float32)
    if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
        return True
    x0, y0, x1, y1 = detection["bbox"]
    return x0 <= x <= x1 and y0 <= y <= y1


def find_detection_at_point(detections, x, y):
    matches = [det for det in detections if point_inside_detection(det, x, y)]
    if not matches:
        return None
    return min(matches, key=lambda item: item["area"])


def detections_overlap(det_a, det_b):
    iou, containment = bbox_iou(det_a["bbox"], det_b["bbox"])
    return iou > 0.18 or containment > 0.64 or centers_too_close(det_a, det_b)


def suppress_duplicates(detections):
    ranked = sorted(
        detections,
        key=lambda det: (det.get("confidence", 0.0), det.get("peak_score", 0.0), det.get("mean_score", 0.0), det["area"]),
        reverse=True,
    )
    kept = []
    for det in ranked:
        if any(detections_overlap(det, existing) for existing in kept):
            continue
        kept.append(det)
    kept.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return kept


def select_detections(detections, target_count, search_mask, max_auto, ctx=None):
    if not detections:
        return []
    if ctx is not None:
        detections = filter_size_constraints(detections, ctx)
        if not detections:
            return []
    detections = suppress_duplicates(detections)
    if target_count is not None and target_count > 0:
        return select_banded_detections(detections, target_count=target_count, search_mask=search_mask)

    ranked = sorted(
        detections,
        key=lambda det: (det.get("confidence", 0.0), det.get("peak_score", 0.0), det["area"]),
        reverse=True,
    )
    confidence_floor = np.percentile([det.get("confidence", 0.0) for det in ranked], 35)
    selected = [det for det in ranked if det.get("confidence", 0.0) >= confidence_floor]
    selected = selected[:max_auto]
    selected.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return selected


def merge_manual_detection(detections, new_detection):
    for idx, existing in enumerate(detections):
        if not detections_overlap(new_detection, existing):
            continue
        if new_detection["confidence"] >= existing["confidence"] * 0.86:
            updated = list(detections)
            updated[idx] = new_detection
            updated.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
            return updated, "replaced nearby detection"
        return detections, "ignored duplicate candidate"
    updated = list(detections)
    updated.append(new_detection)
    updated.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return updated, "added detection"


def local_detection_at_point(ctx, x, y, backend="classic-click"):
    score_map = ctx["hold_score"]
    search_mask = ctx["search_mask"]
    height, width = search_mask.shape
    x = int(round(x))
    y = int(round(y))
    if x < 0 or y < 0 or x >= width or y >= height or search_mask[y, x] == 0:
        return None

    radius = max(24, int(round(min(height, width) * 0.035)))
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + radius + 1)
    y1 = min(height, y + radius + 1)
    roi_score = score_map[y0:y1, x0:x1]
    roi_mask = search_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(roi_mask) == 0:
        return None

    seed_score = float(score_map[y, x])
    thresholds = [
        max(0.07, min(seed_score * 0.58, np.percentile(roi_score[roi_mask], 54))),
        max(0.07, np.percentile(roi_score[roi_mask], 48)),
        max(0.06, np.percentile(roi_score[roi_mask], 42)),
    ]

    best = None
    seed_local = (x - x0, y - y0)
    for threshold in thresholds:
        candidate_roi = ((roi_score >= threshold) & roi_mask).astype(np.uint8) * 255
        candidate_roi = cv2.morphologyEx(
            candidate_roi,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        candidate_roi = cv2.morphologyEx(
            candidate_roi,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        if candidate_roi[seed_local[1], seed_local[0]] == 0:
            cv2.circle(candidate_roi, seed_local, 3, 255, thickness=cv2.FILLED)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(candidate_roi, connectivity=8)
        label = labels[seed_local[1], seed_local[0]]
        if label == 0 and num_labels > 1:
            label = 1 + np.argmax([np.count_nonzero(labels == idx) for idx in range(1, num_labels)])
        if label == 0:
            continue

        region = np.zeros_like(search_mask, dtype=np.uint8)
        region[y0:y1, x0:x1] = np.where(labels == label, 255, 0).astype(np.uint8)
        det = region_to_detection(region, score_map, search_mask)
        if det is None:
            continue
        det = normalize_detection(det, backend)
        if not passes_size_constraints(det, ctx):
            continue
        det["source"] = "manual-click"
        if best is None or det["confidence"] > best["confidence"]:
            best = det
    return best


def build_seed_points(ctx, target_count, max_seeds):
    score_map = ctx["hold_score"]
    search_mask = ctx["search_mask"]
    default_target = target_count or 92
    params = {
        "base_percentile": 58,
        "peak_percentile": 66,
        "min_distance": 7,
        "max_points": max(max_seeds, default_target * 3),
    }
    proposal = propose_detections(score_map, search_mask, params)
    lightweight_selected = select_detections(
        [normalize_detection(det, "classic-seed") for det in proposal["detections"]],
        target_count=min(default_target + 40, max(max_seeds, default_target * 2)),
        search_mask=search_mask,
        max_auto=max_seeds,
        ctx=ctx,
    )
    _, peaks = pick_peak_points(
        score_map,
        search_mask,
        peak_percentile=67,
        min_distance=6,
        max_points=max(max_seeds, default_target * 4),
    )

    seeds = []
    for det in lightweight_selected:
        cx, cy = det["center"]
        seeds.append({"x": int(round(cx)), "y": int(round(cy)), "priority": 1.6 + det["confidence"], "source": "classic"})
    for score, x, y in peaks:
        seeds.append({"x": int(x), "y": int(y), "priority": 0.8 + float(score), "source": "peak"})

    selected = []
    min_distance_sq = 25.0
    for seed in sorted(seeds, key=lambda item: item["priority"], reverse=True):
        if any((seed["x"] - old["x"]) ** 2 + (seed["y"] - old["y"]) ** 2 < min_distance_sq for old in selected):
            continue
        selected.append(seed)
        if len(selected) >= max_seeds:
            break
    return selected


def detection_from_mask(mask, confidence, ctx, backend, source="mask", extra=None):
    raw_mask = np.asarray(mask).astype(bool)
    search_bool = ctx["search_mask"] > 0
    region_bool = raw_mask & search_bool
    area = int(np.count_nonzero(region_bool))
    raw_area = int(np.count_nonzero(raw_mask))
    if area == 0 or raw_area == 0:
        return None

    inside_ratio = area / float(raw_area)
    if inside_ratio < 0.70:
        return None

    region = region_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    search_area = max(int(np.count_nonzero(ctx["search_mask"])), 1)
    min_area = max(14, search_area // 30000)
    max_area = search_area // 5
    if area < min_area or area > max_area or min(w, h) < 3:
        return None

    bbox_area = max(w * h, 1)
    fill_ratio = area / float(bbox_area)
    aspect_ratio = max(w, h) / float(max(1, min(w, h)))
    contour_area = max(cv2.contourArea(contour), 1.0)
    hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    solidity = float(contour_area / hull_area)
    compactness = float((4.0 * np.pi * contour_area) / (perimeter * perimeter))
    if fill_ratio < 0.055:
        return None
    if aspect_ratio > 8.0 and fill_ratio < 0.22:
        return None
    if solidity < 0.12 and fill_ratio < 0.18:
        return None

    scores = ctx["hold_score"][region_bool]
    mean_score = float(scores.mean()) if scores.size else 0.0
    peak_score = float(scores.max()) if scores.size else 0.0
    if peak_score < 0.10:
        return None

    center_x, center_y = contour_center(contour)
    size_score = np.clip(np.sqrt(area) / 22.0, 0.0, 1.0)
    score = (
        0.28 * float(confidence)
        + 0.24 * peak_score
        + 0.16 * mean_score
        + 0.10 * np.clip(fill_ratio, 0.0, 1.0)
        + 0.08 * np.clip(solidity, 0.0, 1.0)
        + 0.06 * np.clip(compactness * 2.0, 0.0, 1.0)
        + 0.08 * size_score
    )
    det = {
        "bbox": (x, y, x + w, y + h),
        "center": (float(center_x), float(center_y)),
        "area": float(area),
        "confidence": float(score),
        "raw_confidence": float(confidence),
        "peak_score": peak_score,
        "mean_score": mean_score,
        "fill_ratio": float(fill_ratio),
        "contour": contour.astype(np.float32),
        "source": source,
    }
    if extra:
        det.update(extra)
    det = normalize_detection(det, backend)
    if not passes_size_constraints(det, ctx):
        return None
    return det


def build_prompt_variants(x, y, width, height):
    variants = []
    for radius in (0, 18, 30):
        coords = [[float(x), float(y)]]
        labels = [1]
        if radius:
            for px, py in ((x - radius, y), (x + radius, y), (x, y - radius), (x, y + radius)):
                if 0 <= px < width and 0 <= py < height:
                    coords.append([float(px), float(py)])
                    labels.append(0)
        variants.append((np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)))
    return variants


def best_sam_detection_at_point(ctx, predictor, x, y, backend):
    height, width = ctx["search_mask"].shape
    best = None
    for point_coords, point_labels in build_prompt_variants(x, y, width, height):
        masks, ious, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=False,
        )
        for mask, iou in zip(masks, ious):
            det = detection_from_mask(mask, float(iou), ctx, backend, source="point-prompt")
            if det is None:
                continue
            if best is None or det["confidence"] > best["confidence"]:
                best = det
    return best


def run_classic(ctx, args):
    target = args.target_count
    score_map = ctx["hold_score"]
    search_mask = ctx["search_mask"]
    if target is None:
        parameter_grid = [
            {"base_percentile": 58, "peak_percentile": 66, "min_distance": 7, "max_points": args.max_seeds}
        ]
    else:
        parameter_grid = [
            {"base_percentile": base, "peak_percentile": peak, "min_distance": dist, "max_points": args.max_seeds}
            for base in (56, 60, 64)
            for peak in (64, 68)
            for dist in (7, 10)
        ]
    best = None
    for params in parameter_grid:
        proposal = propose_detections(score_map, search_mask, params)
        candidates = [normalize_detection(det, "classic") for det in proposal["detections"]]
        selected = select_detections(candidates, target, search_mask, args.max_auto_detections, ctx=ctx)
        if target is None:
            count_score = -abs(len(selected) - min(len(candidates), args.max_auto_detections)) * 0.01
        else:
            count_score = -abs(len(selected) - target)
        confidence_score = np.mean([det["confidence"] for det in selected]) if selected else 0.0
        score = count_score + confidence_score
        if best is None or score > best["score"]:
            best = {"detections": selected, "params": params, "score": score, "raw_candidates": len(candidates)}

    ctx["backend_state"] = {"backend": "classic", "params": best["params"], "raw_candidates": best["raw_candidates"]}
    return best["detections"]


def import_sam_modules(backend):
    import torch

    patch_torch_numpy_compat(torch)
    if backend == "mobile-sam":
        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("Install MobileSAM with: pip install git+https://github.com/ChaoningZhang/MobileSAM.git") from exc
    else:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError("Install Segment Anything with: pip install git+https://github.com/facebookresearch/segment-anything.git") from exc
    patch_sam_predictor_set_image(torch, SamPredictor)
    return torch, SamPredictor, sam_model_registry


def run_sam_like(ctx, args, backend):
    torch, SamPredictor, sam_model_registry = import_sam_modules(backend)
    model_type = args.model_type
    checkpoint = args.checkpoint
    if backend == "sam-vit-b":
        model_type = "vit_b"
        checkpoint = checkpoint or str(MODELS_DIR / "sam_vit_b_01ec64.pth")
    elif backend == "mobile-sam":
        model_type = args.model_type or "vit_t"

    if not checkpoint:
        raise RuntimeError(f"{backend} requires --checkpoint.")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = pick_torch_device(torch)
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)
    predictor.set_image(ctx["image_small"])

    seeds = build_seed_points(ctx, args.target_count, args.max_seeds)
    detections = []
    for idx, seed in enumerate(seeds, start=1):
        det = best_sam_detection_at_point(ctx, predictor, seed["x"], seed["y"], backend)
        if det is not None:
            det["seed"] = (seed["x"], seed["y"])
            detections.append(det)
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(seeds)} seeds, candidates: {len(detections)}", flush=True)

    ctx["backend_state"] = {
        "backend": backend,
        "device": device,
        "model_type": model_type,
        "checkpoint": checkpoint,
        "predictor": predictor,
        "seed_count": len(seeds),
        "raw_candidates": len(detections),
    }
    return select_detections(detections, args.target_count, ctx["search_mask"], args.max_auto_detections, ctx=ctx)


def load_efficient_sam_model(args):
    import torch

    patch_torch_numpy_compat(torch)
    try:
        from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
    except ImportError as exc:
        raise RuntimeError(
            "Install EfficientSAM from its repo, for example: "
            "pip install git+https://github.com/yformer/EfficientSAM.git"
        ) from exc

    builder = build_efficient_sam_vits if args.efficient_variant == "s" else build_efficient_sam_vitt
    model = None
    if args.checkpoint:
        try:
            model = builder(checkpoint=args.checkpoint)
        except TypeError:
            model = builder()
            state = torch.load(args.checkpoint, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
    else:
        model = builder()
    return torch, model


def efficient_sam_detection_at_point(ctx, model, torch, image_tensor, x, y, backend="efficient-sam"):
    height, width = ctx["search_mask"].shape
    device = next(model.parameters()).device
    best = None
    with torch.no_grad():
        for point_coords_np, point_labels_np in build_prompt_variants(x, y, width, height):
            input_points = torch.tensor(point_coords_np, dtype=torch.float32, device=device)[None, None, :, :]
            input_labels = torch.tensor(point_labels_np, dtype=torch.int64, device=device)[None, None, :]
            predicted_logits, predicted_iou = model(image_tensor, input_points, input_labels)
            order = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, order, dim=2)
            predicted_logits = torch.take_along_dim(predicted_logits, order[..., None, None], dim=2)
            logits = predicted_logits[0, 0]
            ious = predicted_iou[0, 0]
            for mask_logits, iou in zip(logits, ious):
                mask = mask_logits.detach().cpu().numpy() >= 0
                if mask.shape != ctx["search_mask"].shape:
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                det = detection_from_mask(mask, float(iou.detach().cpu()), ctx, backend, source="point-prompt")
                if det is None:
                    continue
                if best is None or det["confidence"] > best["confidence"]:
                    best = det
    return best


def run_efficient_sam(ctx, args):
    torch, model = load_efficient_sam_model(args)
    device = pick_torch_device(torch)
    model.to(device=device)
    model.eval()
    image_tensor = torch.from_numpy(ctx["image_small"].astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1)[None, :, :, :].to(device)

    seeds = build_seed_points(ctx, args.target_count, args.max_seeds)
    detections = []
    for idx, seed in enumerate(seeds, start=1):
        det = efficient_sam_detection_at_point(ctx, model, torch, image_tensor, seed["x"], seed["y"])
        if det is not None:
            detections.append(det)
        if idx % 20 == 0:
            print(f"Processed {idx}/{len(seeds)} seeds, candidates: {len(detections)}", flush=True)

    ctx["backend_state"] = {
        "backend": "efficient-sam",
        "device": device,
        "variant": args.efficient_variant,
        "seed_count": len(seeds),
        "raw_candidates": len(detections),
        "model": model,
        "torch": torch,
        "image_tensor": image_tensor,
    }
    return select_detections(detections, args.target_count, ctx["search_mask"], args.max_auto_detections, ctx=ctx)


def run_ultralytics(ctx, args, backend):
    try:
        from ultralytics import FastSAM, YOLO
    except ImportError as exc:
        raise RuntimeError("Install Ultralytics with: pip install ultralytics") from exc

    model_path = args.yolo_model
    if backend == "fastsam":
        model_path = model_path or str(MODELS_DIR / "FastSAM-s.pt")
        model = FastSAM(model_path)
    else:
        model_path = model_path or "yolo11n-seg.pt"
        model = YOLO(model_path)

    kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if backend == "fastsam":
        kwargs["retina_masks"] = True

    results = model(ctx["image_small"], **kwargs)
    result = results[0]
    detections = []
    raw_masks = []
    if getattr(result, "masks", None) is not None and result.masks is not None:
        mask_data = result.masks.data
        masks = mask_data.detach().cpu().numpy() if hasattr(mask_data, "detach") else np.asarray(mask_data)
        confidences = []
        if getattr(result, "boxes", None) is not None and result.boxes is not None:
            boxes_conf = getattr(result.boxes, "conf", None)
            if boxes_conf is not None:
                confidences = boxes_conf.detach().cpu().numpy().tolist()
        for idx, mask in enumerate(masks):
            if mask.shape != ctx["search_mask"].shape:
                mask = cv2.resize(mask.astype(np.float32), ctx["search_mask"].shape[::-1], interpolation=cv2.INTER_LINEAR) >= 0.5
            conf = float(confidences[idx]) if idx < len(confidences) else 0.65
            raw_masks.append({"mask": mask.astype(bool), "confidence": conf})
            det = detection_from_mask(mask, conf, ctx, backend, source="ultralytics")
            if det is not None:
                detections.append(det)

    ctx["backend_state"] = {
        "backend": backend,
        "model_path": model_path,
        "raw_candidates": len(detections),
        "raw_masks": raw_masks,
    }
    return select_detections(detections, args.target_count, ctx["search_mask"], args.max_auto_detections, ctx=ctx)


def detect_at_click(ctx, detections, x_small, y_small):
    backend = ctx["backend_state"].get("backend", "classic")
    if backend in {"sam", "sam-vit-b", "mobile-sam"}:
        predictor = ctx["backend_state"]["predictor"]
        det = best_sam_detection_at_point(ctx, predictor, x_small, y_small, backend)
        return det or local_detection_at_point(ctx, x_small, y_small, backend=f"{backend}-local-click")

    if backend == "efficient-sam":
        state = ctx["backend_state"]
        det = efficient_sam_detection_at_point(
            ctx,
            state["model"],
            state["torch"],
            state["image_tensor"],
            x_small,
            y_small,
        )
        return det or local_detection_at_point(ctx, x_small, y_small, backend="efficient-sam-local-click")

    if backend in {"fastsam", "yolo-seg"}:
        best = None
        for item in ctx["backend_state"].get("raw_masks", []):
            mask = item["mask"]
            x = int(round(x_small))
            y = int(round(y_small))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                det = detection_from_mask(mask, item["confidence"], ctx, backend, source="click-selected-mask")
                if det is not None and (best is None or det["confidence"] > best["confidence"]):
                    best = det
        return best or local_detection_at_point(ctx, x_small, y_small, backend=f"{backend}-local-click")

    return local_detection_at_point(ctx, x_small, y_small, backend="classic-click")


def interactive_edit(ctx, detections_small):
    if "agg" in plt.get_backend().lower():
        print("Interactive edit skipped because the Matplotlib backend is non-interactive.")
        return detections_small

    fig, ax = plt.subplots(figsize=(7, 10))
    polygon = np.vstack([ctx["corners"], ctx["corners"][0]])
    search_contour = np.round(ctx["corners_small"]).astype(np.float32).reshape(-1, 1, 2)
    scale = ctx["scale"]
    state = {"detections": list(detections_small)}

    def redraw():
        ax.clear()
        ax.imshow(ctx["image_rgb"])
        ax.plot(polygon[:, 0], polygon[:, 1], color="#00c7be", linewidth=2.0)
        for idx, det_full in enumerate(detections_to_full(state["detections"], scale), start=1):
            contour = det_full["contour"]
            x0, y0, x1, y1 = det_full["bbox"]
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ff3b30", linewidth=1.4)
            rect = plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=0.9,
                edgecolor="#ff3b30",
                facecolor="none",
                alpha=0.35,
            )
            ax.add_patch(rect)
            ax.text(
                x0,
                max(y0 - 7, 10),
                str(idx),
                color="white",
                fontsize=9,
                bbox={"facecolor": "#ff3b30", "edgecolor": "none", "pad": 1.1},
            )
        ax.set_title(
            "Click a detected segment to delete it. Click a missed hold to detect it. "
            f"Current holds: {len(state['detections'])}. Press Enter or q when done."
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        x_full = float(event.xdata)
        y_full = float(event.ydata)
        x_small = x_full / scale
        y_small = y_full / scale

        hit = find_detection_at_point(state["detections"], x_small, y_small)
        if hit is not None:
            state["detections"] = [det for det in state["detections"] if det is not hit]
            print(f"Removed detection near ({x_full:.1f}, {y_full:.1f}). Count: {len(state['detections'])}")
            redraw()
            return

        if cv2.pointPolygonTest(search_contour, (float(x_small), float(y_small)), False) < 0:
            print(f"Ignored click outside the selected wall at ({x_full:.1f}, {y_full:.1f}).")
            return

        new_det = detect_at_click(ctx, state["detections"], x_small, y_small)
        if new_det is None:
            print(f"No valid hold segment found near ({x_full:.1f}, {y_full:.1f}).")
            return

        state["detections"], message = merge_manual_detection(state["detections"], new_det)
        print(f"{message.capitalize()} near ({x_full:.1f}, {y_full:.1f}). Count: {len(state['detections'])}")
        redraw()

    def onkey(event):
        if event.key in {"enter", "return", "q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    redraw()
    plt.show()
    return state["detections"]


def default_output_path(image_path, backend):
    image_file = Path(image_path)
    safe_backend = backend.replace("-", "_")
    return str(OUTPUTS_DIR / "annotations" / f"{image_file.stem}_{safe_backend}_detected_holds.json")


def save_detected_holds(image_path, ctx, detections_small, output_path):
    detections = detections_to_full(detections_small, ctx["scale"])
    payload = {
        "image": image_path,
        "backend": ctx["backend_state"].get("backend", ""),
        "model": {
            key: value
            for key, value in ctx["backend_state"].items()
            if key in {"model_type", "checkpoint", "device", "model_path", "variant", "params", "seed_count", "raw_candidates"}
        },
        "scale": float(ctx["scale"]),
        "corner_points": [[float(x), float(y)] for x, y in ctx["corners"]],
        "size_constraints": ctx.get("size_constraints"),
        "target_count": len(detections),
        "detections": [],
    }
    for idx, det in enumerate(detections, start=1):
        payload["detections"].append(
            {
                "id": idx,
                "bbox": [int(v) for v in det["bbox"]],
                "center": [float(det["center"][0]), float(det["center"][1])],
                "area": float(det["area"]),
                "confidence": float(det["confidence"]),
                "backend": det.get("backend", ""),
                "source": det.get("source", ""),
                "contour": [[float(pt[0][0]), float(pt[0][1])] for pt in det["contour"]],
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved detected holds to {output_path}")


def plot_results(ctx, detections_small, output_path):
    detections = detections_to_full(detections_small, ctx["scale"])
    fig, ax = plt.subplots(figsize=(7, 10))
    polygon = np.vstack([ctx["corners"], ctx["corners"][0]])
    ax.imshow(ctx["image_rgb"])
    ax.plot(polygon[:, 0], polygon[:, 1], color="#00c7be", linewidth=2.0)
    for det in detections:
        contour = det["contour"]
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ff3b30", linewidth=1.4)
    backend = ctx["backend_state"].get("backend", "")
    raw_candidates = ctx["backend_state"].get("raw_candidates", "?")
    ax.set_title(f"{backend}: {len(detections)} holds | raw candidates: {raw_candidates}")
    ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shared climbing-hold detection workflow: corners, auto-detect, click refine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image", nargs="?", default=str(IMAGES_DIR / "IMG_1637.jpeg"), help="Path to the climbing wall image.")
    parser.add_argument("--backend", choices=BACKENDS, default="classic", help="Detection backend.")
    parser.add_argument("--target-count", type=int, help="Optional expected hold count.")
    parser.add_argument("--max-height", type=int, default=1400, help="Resize height for processing.")
    parser.add_argument("--max-seeds", type=int, default=180, help="Maximum point prompts/seeds for promptable backends.")
    parser.add_argument("--max-auto-detections", type=int, default=140, help="Cap when no target count is given.")
    parser.add_argument("--checkpoint", help="SAM/MobileSAM/EfficientSAM checkpoint path.")
    parser.add_argument("--model-type", help="SAM model type, e.g. vit_b, vit_l, vit_t.")
    parser.add_argument("--efficient-variant", choices=("ti", "s"), default="ti", help="EfficientSAM variant.")
    parser.add_argument("--yolo-model", help="Ultralytics model path, e.g. models/FastSAM-s.pt or a custom hold best.pt.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Ultralytics inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Ultralytics confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="Ultralytics IoU threshold.")
    parser.add_argument(
        "--size-reference-json",
        default=DEFAULT_SIZE_REFERENCE,
        help="Truth JSON used to derive min/max hold size constraints.",
    )
    parser.add_argument("--no-size-filter", action="store_true", help="Disable truth-derived hold size filtering.")
    parser.add_argument(
        "--min-size-tolerance",
        type=float,
        default=1.0,
        help="Lower-bound tolerance for truth-derived size constraints.",
    )
    parser.add_argument(
        "--max-hold-scale",
        type=float,
        default=1.5,
        help="Largest allowed hold scale relative to the truth set. Area uses this value squared.",
    )
    parser.add_argument("--save-holds", help="Output JSON path.")
    parser.add_argument("--save-plot", help="Optional debug plot path.")
    parser.add_argument("--no-edit", action="store_true", help="Skip final interactive click correction.")
    parser.add_argument(
        "--corners",
        nargs=8,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        help="Optional four wall corners to skip interactive corner clicking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    backend = args.backend
    if backend == "sam" and args.model_type is None:
        args.model_type = "vit_l"
    if backend == "sam-vit-b":
        args.model_type = "vit_b"
        if args.checkpoint is None:
            args.checkpoint = str(MODELS_DIR / "sam_vit_b_01ec64.pth")
    if backend == "mobile-sam" and args.model_type is None:
        args.model_type = "vit_t"

    image_rgb = load_image(args.image)
    if args.corners is not None:
        corners = order_corners(np.asarray(args.corners, dtype=np.float32).reshape(4, 2))
    else:
        corners = collect_corners_interactively(image_rgb)

    size_constraints = None
    if not args.no_size_filter:
        size_constraints = load_size_constraints(
            args.size_reference_json,
            min_tolerance=args.min_size_tolerance,
            max_linear_scale=args.max_hold_scale,
        )

    ctx = build_context(image_rgb, corners, args.max_height, size_constraints=size_constraints)
    print(f"Size filter: {describe_size_constraints(size_constraints)}")

    if backend == "classic":
        detections_small = run_classic(ctx, args)
    elif backend in {"sam", "sam-vit-b", "mobile-sam"}:
        detections_small = run_sam_like(ctx, args, backend)
    elif backend == "efficient-sam":
        detections_small = run_efficient_sam(ctx, args)
    elif backend in {"fastsam", "yolo-seg"}:
        detections_small = run_ultralytics(ctx, args, backend)
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")

    print(f"Backend: {ctx['backend_state'].get('backend', backend)}")
    print(f"Initial detections: {len(detections_small)}")
    if not args.no_edit:
        detections_small = interactive_edit(ctx, detections_small)

    print(f"Final detections: {len(detections_small)}")
    for idx, det in enumerate(detections_to_full(detections_small, ctx["scale"]), start=1):
        x0, y0, x1, y1 = det["bbox"]
        cx, cy = det["center"]
        print(f"{idx:03d}: bbox=({x0}, {y0}) -> ({x1}, {y1}), center=({cx:.1f}, {cy:.1f}), area~{det['area']:.0f}px")

    holds_output = args.save_holds or default_output_path(args.image, ctx["backend_state"].get("backend", backend))
    save_detected_holds(args.image, ctx, detections_small, holds_output)
    if args.save_plot:
        plot_results(ctx, detections_small, args.save_plot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
