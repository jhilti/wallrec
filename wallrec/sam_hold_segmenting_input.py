import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from auto_hold_detector import bbox_iou, build_hold_score, centers_too_close

_ORIGINAL_TORCH_AS_TENSOR = torch.as_tensor
_ORIGINAL_TENSOR_NUMPY = torch.Tensor.numpy
_ORIGINAL_TENSOR_ARRAY = torch.Tensor.__array__


def patch_torch_numpy_compat():
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
            dtype = kwargs.pop("dtype", None)
            if dtype is None:
                dtype = dtype_map.get(data.dtype, None)

            if data.dtype == np.uint8 and data.flags["C_CONTIGUOUS"]:
                tensor = torch.frombuffer(bytearray(data.tobytes()), dtype=torch.uint8).clone().reshape(data.shape)
                if dtype is not None and dtype != torch.uint8:
                    tensor = tensor.to(dtype=dtype)
            else:
                tensor = torch.tensor(data.tolist(), dtype=dtype)

            if device is not None:
                tensor = tensor.to(device)
            return tensor

        return _ORIGINAL_TORCH_AS_TENSOR(data, *args, **kwargs)

    def patched_numpy(self):
        return np.asarray(self.detach().cpu().tolist())

    def patched_array(self, dtype=None):
        array = np.asarray(self.detach().cpu().tolist())
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    torch.as_tensor = patched_as_tensor
    torch.Tensor.numpy = patched_numpy
    torch.Tensor.__array__ = patched_array


def patch_predictor_set_image():
    def patched_set_image(self, image, image_format="RGB"):
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.frombuffer(bytearray(input_image.tobytes()), dtype=torch.uint8).clone()
        input_image_torch = input_image_torch.reshape(input_image.shape).to(self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.set_torch_image(input_image_torch, image.shape[:2])

    SamPredictor.set_image = patched_set_image


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path):
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def resize_for_detection(image, max_height=1400):
    height, width = image.shape[:2]
    scale = max(1.0, height / float(max_height))
    if scale <= 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(
        image,
        (int(round(width / scale)), int(round(height / scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def order_corners(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    sums = points.sum(axis=1)
    start_idx = np.argmin(sums)
    points = np.roll(points, -start_idx, axis=0)
    return points


def collect_corners_interactively(image):
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.imshow(image)
    ax.set_title("Click the 4 wall corners, roughly clockwise. Window closes after the 4th click.")
    ax.axis("off")

    corners = []
    marker_plot = [None]

    def redraw():
        if marker_plot[0] is not None:
            marker_plot[0].remove()
        pts = np.asarray(corners, dtype=np.float32)
        if len(pts) > 0:
            marker_plot[0] = ax.scatter(pts[:, 0], pts[:, 1], c="#ff3b30", s=55)
        for idx, (x, y) in enumerate(corners, start=1):
            ax.text(x + 8, y - 8, str(idx), color="white", fontsize=11, bbox={"facecolor": "#ff3b30", "edgecolor": "none", "pad": 1.5})
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is None:
            return
        corners.append([float(event.xdata), float(event.ydata)])
        redraw()
        if len(corners) == 4:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    if len(corners) != 4:
        raise RuntimeError("Expected exactly 4 corner clicks.")
    return order_corners(corners)


def polygon_mask(shape, corners):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 255)
    return mask


def pick_seed_points(score_map, search_mask, min_distance=9, threshold_percentile=68, max_points=180):
    search_bool = search_mask > 0
    if np.count_nonzero(search_bool) == 0:
        return []

    threshold = max(0.12, np.percentile(score_map[search_bool], threshold_percentile))
    smoothed = cv2.GaussianBlur(score_map, (0, 0), 2.0)
    kernel_size = 2 * min_distance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(smoothed, kernel)
    peak_mask = (smoothed >= threshold) & search_bool & (smoothed >= dilated - 1e-6)

    ys, xs = np.where(peak_mask)
    candidates = sorted(
        [(float(smoothed[y, x]), int(x), int(y)) for y, x in zip(ys, xs)],
        reverse=True,
    )

    selected = []
    min_distance_sq = float(min_distance * min_distance)
    for score, x, y in candidates:
        if any((x - sx) ** 2 + (y - sy) ** 2 < min_distance_sq for _, sx, sy in selected):
            continue
        selected.append((score, x, y))
        if len(selected) >= max_points:
            break

    return selected


def detection_from_mask(mask, sam_iou, hold_score, search_mask):
    raw_mask = mask.astype(bool)
    search_bool = search_mask > 0
    region_bool = raw_mask & search_bool
    raw_area = int(np.count_nonzero(raw_mask))
    area = int(np.count_nonzero(region_bool))
    if area == 0 or raw_area == 0:
        return None

    inside_ratio = area / float(raw_area)
    if inside_ratio < 0.72:
        return None

    region = region_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(w * h, 1)
    fill_ratio = area / float(bbox_area)
    aspect_ratio = max(w, h) / float(max(1, min(w, h)))
    contour_area = max(cv2.contourArea(contour), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    solidity = float(contour_area / hull_area)
    compactness = float((4.0 * np.pi * contour_area) / (perimeter * perimeter))

    search_area = max(int(np.count_nonzero(search_mask)), 1)
    min_area = max(16, search_area // 7000)
    max_area = search_area // 8
    if area < min_area or area > max_area:
        return None
    if min(w, h) < 4:
        return None
    if fill_ratio < 0.08:
        return None
    if aspect_ratio > 7.0 and fill_ratio < 0.30:
        return None
    if solidity < 0.20 and fill_ratio < 0.20:
        return None

    scores = hold_score[region_bool]
    mean_score = float(scores.mean()) if scores.size else 0.0
    peak_score = float(scores.max()) if scores.size else 0.0
    if peak_score < 0.18:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        center_x = x + w / 2.0
        center_y = y + h / 2.0
    else:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

    size_score = np.clip(np.sqrt(area) / 24.0, 0.0, 1.0)
    confidence = (
        0.32 * sam_iou
        + 0.22 * peak_score
        + 0.16 * mean_score
        + 0.10 * np.clip(fill_ratio, 0.0, 1.0)
        + 0.08 * np.clip(solidity, 0.0, 1.0)
        + 0.06 * np.clip(compactness * 2.0, 0.0, 1.0)
        + 0.06 * size_score
    )

    return {
        "bbox": (x, y, x + w, y + h),
        "area": area,
        "center": (center_x, center_y),
        "confidence": float(confidence),
        "sam_iou": float(sam_iou),
        "peak_score": peak_score,
        "mean_score": mean_score,
        "fill_ratio": fill_ratio,
        "region_mask": region,
        "contour": contour.copy(),
    }


def choose_best_mask_for_seed(predictor, x, y, hold_score, search_mask):
    masks, ious, _ = predictor.predict(
        point_coords=np.array([[x, y]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
        return_logits=False,
    )

    best = None
    for mask, sam_iou in zip(masks, ious):
        detection = detection_from_mask(mask, float(sam_iou), hold_score, search_mask)
        if detection is None:
            continue
        if best is None or detection["confidence"] > best["confidence"]:
            best = detection
    return best


def suppress_duplicates(detections):
    ranked = sorted(
        detections,
        key=lambda det: (det["confidence"], det["sam_iou"], det["peak_score"], det["area"]),
        reverse=True,
    )
    kept = []
    for det in ranked:
        duplicate = False
        for existing in kept:
            iou, containment = bbox_iou(det["bbox"], existing["bbox"])
            if iou > 0.20 or containment > 0.68 or centers_too_close(det, existing):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def select_final_detections(detections, target_count, image_height):
    deduped = suppress_duplicates(detections)

    lower_band_start = 0.78 * image_height
    kickboard_start = 0.90 * image_height
    desired_kickboard = max(4, target_count // 18)
    desired_lower = max(20, target_count // 4)

    kickboard = [det for det in deduped if det["center"][1] >= kickboard_start]
    lower = [det for det in deduped if lower_band_start <= det["center"][1] < kickboard_start]
    upper = [det for det in deduped if det["center"][1] < lower_band_start]

    selected = []
    for pool, limit in [
        (kickboard, desired_kickboard),
        (lower, desired_lower),
        (upper, target_count),
        (lower, target_count),
        (kickboard, target_count),
    ]:
        for det in pool:
            duplicate = False
            for existing in selected:
                iou, containment = bbox_iou(det["bbox"], existing["bbox"])
                if iou > 0.20 or containment > 0.68 or centers_too_close(det, existing):
                    duplicate = True
                    break
            if duplicate:
                continue

            kick_count = sum(item["center"][1] >= kickboard_start for item in selected)
            lower_count = sum(lower_band_start <= item["center"][1] < kickboard_start for item in selected)
            if pool is kickboard and kick_count >= limit:
                break
            if pool is lower and lower_count >= limit:
                break

            selected.append(det)
            if len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        for det in deduped:
            duplicate = False
            for existing in selected:
                iou, containment = bbox_iou(det["bbox"], existing["bbox"])
                if iou > 0.20 or containment > 0.68 or centers_too_close(det, existing):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
            if len(selected) >= target_count:
                break

    selected = sorted(selected, key=lambda item: item["confidence"], reverse=True)[:target_count]
    selected.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return selected


def scale_detection(detection, scale):
    contour = detection["contour"].astype(np.float32).copy()
    contour[:, 0, 0] *= scale
    contour[:, 0, 1] *= scale
    x0, y0, x1, y1 = detection["bbox"]
    cx, cy = detection["center"]
    return {
        "bbox": (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        ),
        "center": (float(cx * scale), float(cy * scale)),
        "area": float(detection["area"] * scale * scale),
        "confidence": detection["confidence"],
        "contour": contour,
    }


def rebuild_result_from_small(result):
    final_mask = np.zeros_like(result["search_mask"])
    for detection in result["detections_small"]:
        final_mask = cv2.bitwise_or(final_mask, detection["region_mask"])

    result["final_mask"] = final_mask
    result["detections"] = [scale_detection(det, result["scale"]) for det in result["detections_small"]]
    return result


def point_inside_detection(detection, x, y):
    contour = detection["contour"].astype(np.float32)
    if cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0:
        return True
    x0, y0, x1, y1 = detection["bbox"]
    return x0 <= x <= x1 and y0 <= y <= y1


def find_detection_at_point(detections_small, x, y):
    matches = [det for det in detections_small if point_inside_detection(det, x, y)]
    if not matches:
        return None
    return min(matches, key=lambda det: det["area"])


def merge_manual_detection(detections_small, new_detection):
    replacement_idx = None
    for idx, existing in enumerate(detections_small):
        iou, containment = bbox_iou(new_detection["bbox"], existing["bbox"])
        if iou > 0.20 or containment > 0.68 or centers_too_close(new_detection, existing):
            if new_detection["confidence"] >= existing["confidence"]:
                replacement_idx = idx
            else:
                return detections_small, False, "ignored duplicate candidate"
            break

    updated = list(detections_small)
    if replacement_idx is not None:
        updated[replacement_idx] = new_detection
        return updated, True, "replaced nearby detection"

    updated.append(new_detection)
    return updated, True, "added detection"


def interactive_edit_detections(image_rgb, corners, result):
    backend = plt.get_backend().lower()
    if "agg" in backend:
        print("Interactive edit skipped because the current Matplotlib backend is non-interactive.")
        return result

    fig, ax = plt.subplots(figsize=(6, 9))
    ordered_corners = order_corners(corners)
    polygon = np.vstack([ordered_corners, ordered_corners[0]])
    search_contour = np.round(result["corners_small"]).astype(np.float32).reshape(-1, 1, 2)

    def redraw():
        ax.clear()
        ax.imshow(image_rgb)
        ax.plot(polygon[:, 0], polygon[:, 1], color="#00c7be", linewidth=2.2)
        for idx, det in enumerate(result["detections"], start=1):
            contour = det["contour"]
            x0, y0, x1, y1 = det["bbox"]
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ff3b30", linewidth=1.5)
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor="#ff3b30", facecolor="none", alpha=0.35)
            ax.add_patch(rect)
            ax.text(x0, max(y0 - 8, 10), str(idx), color="white", fontsize=10, bbox={"facecolor": "#ff3b30", "edgecolor": "none", "pad": 1.2})
        ax.set_title(
            "Click false detections to remove them.\n"
            "Click missed holds to ask SAM for a new mask there.\n"
            f"Current detections: {len(result['detections'])}. Press Enter, q, or close window when done."
        )
        ax.axis("off")
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        x_full = float(event.xdata)
        y_full = float(event.ydata)
        x_small = x_full / result["scale"]
        y_small = y_full / result["scale"]

        hit = find_detection_at_point(result["detections_small"], x_small, y_small)
        if hit is not None:
            result["detections_small"] = [det for det in result["detections_small"] if det is not hit]
            rebuild_result_from_small(result)
            print(f"Removed detection near ({x_full:.1f}, {y_full:.1f}). New count: {len(result['detections'])}")
            redraw()
            return

        if cv2.pointPolygonTest(search_contour, (x_small, y_small), False) < 0:
            print(f"Ignored click outside search space at ({x_full:.1f}, {y_full:.1f}).")
            return

        detection = choose_best_mask_for_seed(
            result["predictor"],
            x_small,
            y_small,
            result["hold_score"],
            result["search_mask"],
        )
        if detection is None:
            print(f"No valid SAM hold mask found near ({x_full:.1f}, {y_full:.1f}).")
            return

        updated, changed, message = merge_manual_detection(result["detections_small"], detection)
        if changed:
            result["detections_small"] = updated
            rebuild_result_from_small(result)
            print(f"{message.capitalize()} near ({x_full:.1f}, {y_full:.1f}). New count: {len(result['detections'])}")
            redraw()
        else:
            print(f"SAM candidate near ({x_full:.1f}, {y_full:.1f}) was {message}.")

    def onkey(event):
        if event.key in {"enter", "return", "q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    redraw()
    plt.show()
    return result


def run_sam_hold_search(
    image_rgb,
    corners,
    checkpoint,
    model_type="vit_l",
    target_count=92,
    max_height=1400,
    max_seeds=180,
    peak_threshold_percentile=68,
    min_distance=9,
):
    patch_torch_numpy_compat()
    patch_predictor_set_image()

    image_small, scale = resize_for_detection(image_rgb, max_height=max_height)
    corners_small = np.asarray(corners, dtype=np.float32) / scale
    search_mask = polygon_mask(image_small.shape, corners_small)
    hold_score = build_hold_score(image_small, search_mask)
    seeds = pick_seed_points(
        hold_score,
        search_mask,
        min_distance=min_distance,
        threshold_percentile=peak_threshold_percentile,
        max_points=max_seeds,
    )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    device = pick_device()
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(image_small)

    detections = []
    for idx, (_, x, y) in enumerate(seeds, start=1):
        detection = choose_best_mask_for_seed(predictor, x, y, hold_score, search_mask)
        if detection is not None:
            detection["seed"] = (x, y)
            detections.append(detection)
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(seeds)} seeds, candidates so far: {len(detections)}")

    final_small = select_final_detections(detections, target_count=target_count, image_height=image_small.shape[0])
    final_mask = np.zeros_like(search_mask)
    for detection in final_small:
        final_mask = cv2.bitwise_or(final_mask, detection["region_mask"])

    return {
        "device": device,
        "scale": scale,
        "image_small": image_small,
        "corners_small": corners_small,
        "search_mask": search_mask,
        "hold_score": hold_score,
        "seed_points": seeds,
        "raw_candidates": len(detections),
        "detections_small": final_small,
        "detections": [scale_detection(det, scale) for det in final_small],
        "final_mask": final_mask,
        "predictor": predictor,
    }


def plot_results(image_rgb, corners, result, image_path, save_path=None):
    fig, ax_overlay = plt.subplots(figsize=(6.5, 10))
    ordered_corners = order_corners(corners)
    polygon = np.vstack([ordered_corners, ordered_corners[0]])

    ax_overlay.imshow(image_rgb)
    ax_overlay.plot(polygon[:, 0], polygon[:, 1], color="#00c7be", linewidth=2.2)
    for idx, det in enumerate(result["detections"], start=1):
        contour = det["contour"]
        x0, y0, x1, y1 = det["bbox"]
        ax_overlay.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ff3b30", linewidth=1.5)
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0, edgecolor="#ff3b30", facecolor="none", alpha=0.35)
        ax_overlay.add_patch(rect)
        ax_overlay.text(x0, max(y0 - 8, 10), str(idx), color="white", fontsize=10, bbox={"facecolor": "#ff3b30", "edgecolor": "none", "pad": 1.2})
    ax_overlay.set_title(
        f"SAM from clicked corners: {len(result['detections'])}\n"
        f"Seeds: {len(result['seed_points'])} | Raw candidates: {result['raw_candidates']} | Device: {result['device']}"
    )
    ax_overlay.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def default_holds_output_path(image_path):
    image_file = Path(image_path)
    return str(image_file.with_name(f"{image_file.stem}_detected_holds.json"))


def save_detected_holds(image_path, corners, result, output_path):
    payload = {
        "image": image_path,
        "device": result["device"],
        "scale": result["scale"],
        "target_count": len(result["detections"]),
        "corner_points": [[float(x), float(y)] for x, y in np.asarray(corners, dtype=np.float32)],
        "detections": [],
    }

    for idx, det in enumerate(result["detections"], start=1):
        contour = det["contour"]
        payload["detections"].append(
            {
                "id": idx,
                "bbox": [int(v) for v in det["bbox"]],
                "center": [float(det["center"][0]), float(det["center"][1])],
                "area": float(det["area"]),
                "confidence": float(det["confidence"]),
                "contour": [[float(pt[0][0]), float(pt[0][1])] for pt in contour],
            }
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved detected holds to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive SAM hold search using 4 clicked wall corners.")
    parser.add_argument("image", nargs="?", default="IMG_1505.jpeg", help="Path to the wall image.")
    parser.add_argument("--checkpoint", default="sam_vit_l_0b3195.pth", help="Path to the SAM checkpoint.")
    parser.add_argument("--model-type", default="vit_l", help="SAM model type.")
    parser.add_argument("--target-count", type=int, default=92, help="Desired number of final holds.")
    parser.add_argument("--max-height", type=int, default=1400, help="Resize height for SAM processing.")
    parser.add_argument("--max-seeds", type=int, default=180, help="Maximum number of SAM seed points.")
    parser.add_argument("--peak-threshold-percentile", type=float, default=68.0, help="Seed-point threshold percentile.")
    parser.add_argument("--min-distance", type=int, default=9, help="Minimum distance between seed points.")
    parser.add_argument("--save", help="Optional path to save the plot.")
    parser.add_argument("--save-holds", help="Optional path for the final detected holds JSON.")
    parser.add_argument("--no-edit", action="store_true", help="Skip the final interactive correction step.")
    parser.add_argument(
        "--corners",
        nargs=8,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        help="Optional 4 corners to skip interactive clicking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_rgb = load_image(args.image)

    if args.corners is not None:
        corners = np.asarray(args.corners, dtype=np.float32).reshape(4, 2)
        corners = order_corners(corners)
    else:
        corners = collect_corners_interactively(image_rgb)

    result = run_sam_hold_search(
        image_rgb,
        corners=corners,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        target_count=args.target_count,
        max_height=args.max_height,
        max_seeds=args.max_seeds,
        peak_threshold_percentile=args.peak_threshold_percentile,
        min_distance=args.min_distance,
    )

    if not args.no_edit:
        print("Opening interactive correction step.")
        result = interactive_edit_detections(image_rgb, corners, result)

    print(f"Image: {args.image}")
    print(f"Device: {result['device']}")
    print(f"Seeds: {len(result['seed_points'])}")
    print(f"Raw SAM candidates: {result['raw_candidates']}")
    print(f"Final detections: {len(result['detections'])}")
    for idx, det in enumerate(result["detections"], start=1):
        x0, y0, x1, y1 = det["bbox"]
        cx, cy = det["center"]
        print(f"{idx:02d}: bbox=({x0}, {y0}) -> ({x1}, {y1}), center=({cx:.1f}, {cy:.1f}), area~{det['area']:.0f}px")

    holds_output_path = args.save_holds or default_holds_output_path(args.image)
    save_detected_holds(args.image, corners, result, holds_output_path)
    plot_results(image_rgb, corners, result, args.image, save_path=args.save)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
