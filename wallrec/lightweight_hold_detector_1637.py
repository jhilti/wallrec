import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from wallrec.auto_hold_detector import bbox_iou, centers_too_close, load_image, normalize_map


def order_corners(points):
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start_idx = np.argmin(points.sum(axis=1))
    return np.roll(points, -start_idx, axis=0)


def polygon_mask(shape, corners):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(corners).astype(np.int32)], 255)
    return mask


def board_vertical_bounds(search_mask):
    ys = np.where(search_mask > 0)[0]
    if ys.size == 0:
        return 0.0, float(search_mask.shape[0] - 1)
    return float(ys.min()), float(ys.max())


def warp_board(image, corners):
    corners = order_corners(corners).astype(np.float32)
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])
    warp_width = int(round(max(top_width, bottom_width)))
    warp_height = int(round(max(left_height, right_height)))
    dst = np.array(
        [
            [0.0, 0.0],
            [warp_width - 1.0, 0.0],
            [warp_width - 1.0, warp_height - 1.0],
            [0.0, warp_height - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    inverse = cv2.getPerspectiveTransform(dst, corners)
    warped = cv2.warpPerspective(image, matrix, (warp_width, warp_height), flags=cv2.INTER_LINEAR)
    return warped, matrix, inverse, dst


def resize_image(image, max_height=1600):
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


def load_ground_truth(path):
    payload = json.loads(Path(path).read_text())
    corners = order_corners(payload["corner_points"])
    detections = payload["detections"]
    return payload, corners, detections


def scale_gt_detections(gt_detections, scale):
    scaled = []
    for det in gt_detections:
        contour = np.asarray(det["contour"], dtype=np.float32).reshape(-1, 1, 2) / scale
        bbox = [int(round(v / scale)) for v in det["bbox"]]
        center = [float(det["center"][0] / scale), float(det["center"][1] / scale)]
        scaled.append(
            {
                "id": det["id"],
                "bbox": tuple(bbox),
                "center": tuple(center),
                "area": float(det["area"] / (scale * scale)),
                "contour": contour,
            }
        )
    return scaled


def transform_points(points, matrix):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix)
    return transformed.reshape(-1, 2)


def build_search_hold_score(image_small, search_mask):
    rgb = image_small.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_small, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image_small, cv2.COLOR_RGB2LAB).astype(np.float32)

    wall_bool = search_mask > 0

    rgb_blur = cv2.GaussianBlur(rgb, (0, 0), 15)
    lab_blur = cv2.GaussianBlur(lab, (0, 0), 19)
    gray_blur = cv2.GaussianBlur(gray, (0, 0), 11)

    color_delta = np.linalg.norm(rgb - rgb_blur, axis=-1)
    lab_delta = np.linalg.norm(lab - lab_blur, axis=-1)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    brightness_delta = np.abs(gray - gray_blur)
    gradient = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = np.abs(gradient)
    edges = cv2.Canny(image_small, 50, 140).astype(np.float32) / 255.0

    gray_u8 = np.uint8(gray * 255.0)
    top_hat = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32) / 255.0
    black_hat = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32) / 255.0

    wood_bias = np.abs(lab[:, :, 1] - 128.0) / 30.0 + np.abs(lab[:, :, 2] - 128.0) / 34.0
    wood_bias = 1.0 - np.clip(wood_bias, 0.0, 1.0)
    unsat_but_textured = np.clip((1.0 - saturation) * (0.65 * brightness_delta + 0.35 * texture), 0.0, 1.0)

    color_delta = normalize_map(color_delta, wall_bool)
    lab_delta = normalize_map(lab_delta, wall_bool)
    saturation = normalize_map(saturation, wall_bool)
    value = normalize_map(value, wall_bool)
    brightness_delta = normalize_map(brightness_delta, wall_bool)
    texture = normalize_map(texture, wall_bool)
    edges = normalize_map(edges, wall_bool)
    top_hat = normalize_map(top_hat, wall_bool)
    black_hat = normalize_map(black_hat, wall_bool)
    wood_bias = normalize_map(wood_bias, wall_bool)
    unsat_but_textured = normalize_map(unsat_but_textured, wall_bool)

    board_ymin, board_ymax = board_vertical_bounds(search_mask)
    row_positions = np.linspace(0.0, 1.0, image_small.shape[0], dtype=np.float32)
    denom = max(board_ymax - board_ymin, 1.0)
    row_positions = np.clip((np.arange(image_small.shape[0], dtype=np.float32) - board_ymin) / denom, 0.0, 1.0)
    row_positions = row_positions[:, None]
    lower_boost = np.clip((row_positions - 0.70) / 0.30, 0.0, 1.0)

    score = (
        0.24 * lab_delta
        + 0.16 * color_delta
        + 0.08 * saturation
        + 0.10 * brightness_delta
        + 0.10 * texture
        + 0.06 * edges
        + 0.09 * top_hat
        + 0.07 * black_hat
        + 0.04 * value
        + 0.03 * wood_bias
        + 0.03 * unsat_but_textured
        + 0.05 * lower_boost
    )
    score *= wall_bool
    return score.astype(np.float32)


def pick_peak_points(score_map, search_mask, peak_percentile, min_distance, max_points):
    smoothed = cv2.GaussianBlur(score_map, (0, 0), 2.1)
    kernel_size = 2 * min_distance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(smoothed, kernel)
    min_distance_sq = float(min_distance * min_distance)

    board_ymin, board_ymax = board_vertical_bounds(search_mask)
    band_ratios = [0.10, 0.21, 0.29, 0.25, 0.15]
    candidates = []

    for band_index, ratio in enumerate(band_ratios):
        band_start = board_ymin + (board_ymax - board_ymin) * (band_index / len(band_ratios))
        band_end = board_ymin + (board_ymax - board_ymin) * ((band_index + 1) / len(band_ratios))
        row_mask = np.zeros(search_mask.shape[:2], dtype=bool)
        y0 = max(0, int(np.floor(band_start)))
        y1 = min(search_mask.shape[0], int(np.ceil(band_end)))
        row_mask[y0:y1, :] = True
        band_bool = (search_mask > 0) & row_mask
        if np.count_nonzero(band_bool) < 16:
            continue

        threshold = max(0.10, np.percentile(smoothed[band_bool], peak_percentile))
        band_peak_mask = band_bool & (smoothed >= threshold) & (smoothed >= dilated - 1e-6)
        ys, xs = np.where(band_peak_mask)
        band_candidates = sorted(
            [(float(smoothed[y, x]), int(x), int(y)) for y, x in zip(ys, xs)],
            reverse=True,
        )

        band_limit = max(8, int(np.ceil(max_points * ratio * 1.4)))
        band_selected = []
        for score, x, y in band_candidates:
            if any((x - sx) ** 2 + (y - sy) ** 2 < min_distance_sq for _, sx, sy in band_selected):
                continue
            band_selected.append((score, x, y))
            if len(band_selected) >= band_limit:
                break
        candidates.extend(band_selected)

    candidates.sort(reverse=True)
    selected = []
    for score, x, y in candidates:
        if any((x - sx) ** 2 + (y - sy) ** 2 < min_distance_sq for _, sx, sy in selected):
            continue
        selected.append((score, x, y))
        if len(selected) >= max_points:
            break
    return smoothed, selected


def split_component_by_peaks(component_mask, peaks_in_component):
    ys, xs = np.where(component_mask > 0)
    if len(peaks_in_component) <= 1:
        return [component_mask]

    coords = np.column_stack((xs, ys)).astype(np.float32)
    peak_coords = np.array([[peak[1], peak[2]] for peak in peaks_in_component], dtype=np.float32)
    distances = ((coords[:, None, :] - peak_coords[None, :, :]) ** 2).sum(axis=2)
    assignments = np.argmin(distances, axis=1)

    regions = []
    for idx in range(len(peaks_in_component)):
        assigned = assignments == idx
        if not np.any(assigned):
            continue
        region = np.zeros_like(component_mask, dtype=np.uint8)
        region[ys[assigned], xs[assigned]] = 255
        regions.append(region)
    return regions


def extract_regions(score_map, search_mask, peaks, base_percentile):
    search_bool = search_mask > 0
    threshold = max(0.08, np.percentile(score_map[search_bool], base_percentile))
    base_mask = (score_map >= threshold).astype(np.uint8) * 255
    base_mask = cv2.bitwise_and(base_mask, search_mask)
    base_mask = cv2.morphologyEx(
        base_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    base_mask = cv2.morphologyEx(
        base_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(base_mask, connectivity=8)
    regions = []
    for label in range(1, num_labels):
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        peaks_in_component = [peak for peak in peaks if labels[peak[2], peak[1]] == label]
        if not peaks_in_component:
            continue
        regions.extend(split_component_by_peaks(component_mask, peaks_in_component))
    return base_mask, regions


def region_to_detection(region, score_map, search_mask):
    search_area = max(int(np.count_nonzero(search_mask)), 1)
    height, width = search_mask.shape
    area = cv2.countNonZero(region)
    min_area = max(10, search_area // 9000)
    max_area = search_area // 7
    if area < min_area or area > max_area:
        return None

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
    border_distance = min(x, y, max(0, width - (x + w)), max(0, height - (y + h)))

    if min(w, h) < 3:
        return None
    if fill_ratio < 0.06:
        return None
    if aspect_ratio > 7.5 and fill_ratio < 0.24:
        return None
    if solidity < 0.14 and fill_ratio < 0.18:
        return None
    region_bool = region > 0
    region_scores = score_map[region_bool]
    mean_score = float(region_scores.mean()) if region_scores.size else 0.0
    peak_score = float(region_scores.max()) if region_scores.size else 0.0
    if peak_score < 0.14:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        center_x = x + w / 2.0
        center_y = y + h / 2.0
    else:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

    size_score = np.clip(np.sqrt(area) / 22.0, 0.0, 1.0)
    confidence = (
        0.32 * peak_score
        + 0.20 * mean_score
        + 0.14 * np.clip(fill_ratio, 0.0, 1.0)
        + 0.10 * np.clip(solidity, 0.0, 1.0)
        + 0.06 * np.clip(compactness * 2.0, 0.0, 1.0)
        + 0.08 * size_score
        + 0.10 * (1.0 - np.clip(abs(aspect_ratio - 1.7) / 3.5, 0.0, 1.0))
    )

    return {
        "bbox": (x, y, x + w, y + h),
        "center": (center_x, center_y),
        "area": area,
        "confidence": float(confidence),
        "peak_score": peak_score,
        "mean_score": mean_score,
        "fill_ratio": fill_ratio,
        "region_mask": region.copy(),
        "contour": contour.copy(),
    }


def propose_detections(score_map, search_mask, params):
    smoothed, peaks = pick_peak_points(
        score_map,
        search_mask,
        peak_percentile=params["peak_percentile"],
        min_distance=params["min_distance"],
        max_points=params["max_points"],
    )
    base_mask, regions = extract_regions(
        smoothed,
        search_mask,
        peaks,
        base_percentile=params["base_percentile"],
    )
    detections = []
    for region in regions:
        det = region_to_detection(region, smoothed, search_mask)
        if det is not None:
            detections.append(det)
    return {
        "score_map": score_map,
        "smoothed_score": smoothed,
        "base_mask": base_mask,
        "peaks": peaks,
        "detections": detections,
    }


def suppress_duplicates(detections):
    ranked = sorted(
        detections,
        key=lambda det: (det["confidence"], det["peak_score"], det["mean_score"], det["area"]),
        reverse=True,
    )
    kept = []
    for det in ranked:
        duplicate = False
        for existing in kept:
            iou, containment = bbox_iou(det["bbox"], existing["bbox"])
            if iou > 0.18 or containment > 0.62 or centers_too_close(det, existing):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def select_final_detections(detections, target_count, search_mask):
    deduped = suppress_duplicates(detections)
    board_ymin, board_ymax = board_vertical_bounds(search_mask)
    board_height = max(board_ymax - board_ymin, 1.0)
    band_edges = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
    band_ratios = [0.10, 0.21, 0.29, 0.25, 0.15]
    band_targets = [int(round(target_count * ratio)) for ratio in band_ratios]
    band_targets[-1] += target_count - sum(band_targets)

    bands = []
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        y0 = board_ymin + start * board_height
        y1 = board_ymin + end * board_height
        bands.append([det for det in deduped if y0 <= det["center"][1] < y1])

    selected = []
    for pool, limit in zip(bands, band_targets):
        band_count = 0
        for det in pool:
            duplicate = False
            for existing in selected:
                iou, containment = bbox_iou(det["bbox"], existing["bbox"])
                if iou > 0.18 or containment > 0.62 or centers_too_close(det, existing):
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(det)
            band_count += 1
            if band_count >= limit:
                break
            if len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        for det in deduped:
            duplicate = False
            for existing in selected:
                iou, containment = bbox_iou(det["bbox"], existing["bbox"])
                if iou > 0.18 or containment > 0.62 or centers_too_close(det, existing):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(det)
            if len(selected) >= target_count:
                break

    selected = sorted(selected, key=lambda item: item["confidence"], reverse=True)[:target_count]
    selected.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return selected


def center_distance(det_a, det_b):
    ax, ay = det_a["center"]
    bx, by = det_b["center"]
    return float(np.hypot(ax - bx, ay - by))


def match_predictions(predictions, gt_detections):
    candidate_pairs = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(gt_detections):
            iou, containment = bbox_iou(pred["bbox"], gt["bbox"])
            dist = center_distance(pred, gt)
            gt_w = gt["bbox"][2] - gt["bbox"][0]
            gt_h = gt["bbox"][3] - gt["bbox"][1]
            pred_w = pred["bbox"][2] - pred["bbox"][0]
            pred_h = pred["bbox"][3] - pred["bbox"][1]
            distance_limit = max(18.0, 0.55 * max(gt_w, gt_h, pred_w, pred_h))
            closeness = max(0.0, 1.0 - dist / distance_limit)
            similarity = 0.62 * max(iou, containment * 0.85) + 0.38 * closeness
            if similarity >= 0.18:
                candidate_pairs.append((similarity, pred_idx, gt_idx))

    candidate_pairs.sort(reverse=True)
    matched_preds = set()
    matched_gts = set()
    matches = []
    for similarity, pred_idx, gt_idx in candidate_pairs:
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matches.append((pred_idx, gt_idx, similarity))

    precision = len(matches) / max(len(predictions), 1)
    recall = len(matches) / max(len(gt_detections), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "matches": matches,
        "matched_pred_indices": matched_preds,
        "matched_gt_indices": matched_gts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_error": abs(len(predictions) - len(gt_detections)),
    }


def evaluate_params(image_small, search_mask, gt_small, params, target_count):
    score_map = build_search_hold_score(image_small, search_mask)
    proposal = propose_detections(score_map, search_mask, params)
    selected = select_final_detections(proposal["detections"], target_count=target_count, search_mask=search_mask)
    metrics = match_predictions(selected, gt_small)
    score = metrics["f1"] - 0.0025 * metrics["count_error"]
    return {
        "proposal": proposal,
        "selected": selected,
        "metrics": metrics,
        "score": score,
        "params": params,
    }


def tune_detector(image_small, search_mask, gt_small, target_count):
    score_map = build_search_hold_score(image_small, search_mask)
    parameter_grid = []
    for base_percentile in [58, 61, 64]:
        for peak_percentile in [66, 69, 72]:
            for min_distance in [7, 9, 11]:
                for max_points in [180, 220]:
                    parameter_grid.append(
                        {
                            "base_percentile": base_percentile,
                            "peak_percentile": peak_percentile,
                            "min_distance": min_distance,
                            "max_points": max_points,
                        }
                    )

    best = None
    for idx, params in enumerate(parameter_grid, start=1):
        proposal = propose_detections(score_map, search_mask, params)
        selected = select_final_detections(proposal["detections"], target_count=target_count, search_mask=search_mask)
        metrics = match_predictions(selected, gt_small)
        result = {
            "proposal": proposal,
            "selected": selected,
            "metrics": metrics,
            "score": metrics["f1"] - 0.0025 * metrics["count_error"],
            "params": params,
        }
        if best is None or result["score"] > best["score"]:
            best = result
            print(
                f"New best {idx}/{len(parameter_grid)}: "
                f"f1={best['metrics']['f1']:.3f}, "
                f"precision={best['metrics']['precision']:.3f}, "
                f"recall={best['metrics']['recall']:.3f}, "
                f"pred={len(best['selected'])}, params={best['params']}"
                ,
                flush=True,
            )
    return best


def scale_detection(det, scale):
    contour = det["contour"].astype(np.float32).copy()
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
        "confidence": det["confidence"],
        "contour": contour,
    }


def unwarp_detection(det, inverse_matrix):
    contour = det["contour"].astype(np.float32).reshape(-1, 1, 2)
    contour = cv2.perspectiveTransform(contour, inverse_matrix)
    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        center_x = x + w / 2.0
        center_y = y + h / 2.0
    else:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
    return {
        "bbox": (x, y, x + w, y + h),
        "center": (float(center_x), float(center_y)),
        "area": float(det["area"]),
        "confidence": det["confidence"],
        "peak_score": det["peak_score"],
        "mean_score": det["mean_score"],
        "contour": contour.astype(np.float32),
        "region_mask": det["region_mask"],
    }


def propose_warped_detections(score_map_warped, search_mask_warped, params, inverse_matrix):
    proposal = propose_detections(score_map_warped, search_mask_warped, params)
    detections = [unwarp_detection(det, inverse_matrix) for det in proposal["detections"]]
    return proposal, detections


def save_detected_holds(image_path, corners, detections, output_path):
    payload = {
        "image": image_path,
        "corner_points": [[float(x), float(y)] for x, y in np.asarray(corners, dtype=np.float32)],
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
                "contour": [[float(pt[0][0]), float(pt[0][1])] for pt in det["contour"]],
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved detected holds to {output_path}")


def plot_comparison(image_rgb, corners, detections, gt_detections, metrics, output_path=None):
    fig, ax = plt.subplots(figsize=(7, 11))
    polygon = np.vstack([corners, corners[0]])
    ax.imshow(image_rgb)
    ax.plot(polygon[:, 0], polygon[:, 1], color="#00c7be", linewidth=2.0)

    matched_pred = metrics["matched_pred_indices"]
    for idx, det in enumerate(detections):
        contour = det["contour"]
        color = "#00c851" if idx in matched_pred else "#ff3b30"
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=1.5)

    gt_unmatched = set(range(len(gt_detections))) - metrics["matched_gt_indices"]
    for idx in gt_unmatched:
        gt = gt_detections[idx]
        contour = np.asarray(gt["contour"], dtype=np.float32).reshape(-1, 1, 2)
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ffd60a", linewidth=1.5)

    ax.set_title(
        f"Lightweight detector vs GT\n"
        f"F1={metrics['f1']:.3f} | Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f} | "
        f"Pred={len(detections)} | GT={len(gt_detections)}"
    )
    ax.axis("off")
    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight hold detector tuned against IMG_1637 ground truth.")
    parser.add_argument("image", nargs="?", default="data/images/IMG_1637.jpeg", help="Image to detect holds in.")
    parser.add_argument("--gt-json", default="data/annotations/IMG_1637_detected_holds.json", help="Ground-truth JSON used for tuning/evaluation.")
    parser.add_argument("--max-height", type=int, default=1600, help="Resize height for tuning and detection.")
    parser.add_argument("--save-plot", default="reports/plots/lightweight_hold_detector_1637.png", help="Path for the comparison plot.")
    parser.add_argument("--save-holds", default="outputs/annotations/IMG_1637_lightweight_detected_holds.json", help="Path for the output detections JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_rgb = load_image(args.image)
    _, corners, gt_detections = load_ground_truth(args.gt_json)

    image_small, scale = resize_image(image_rgb, max_height=args.max_height)
    corners_small = corners / scale
    search_mask = polygon_mask(image_small.shape, corners_small)
    gt_small = scale_gt_detections(gt_detections, scale)

    best = tune_detector(image_small, search_mask, gt_small, target_count=len(gt_detections))
    detections_small = best["selected"]
    detections_full = [scale_detection(det, scale) for det in detections_small]
    metrics_full = match_predictions(detections_full, gt_detections)

    print("\nBest params:", best["params"])
    print(
        f"Final metrics: f1={metrics_full['f1']:.3f}, "
        f"precision={metrics_full['precision']:.3f}, "
        f"recall={metrics_full['recall']:.3f}, "
        f"pred={len(detections_full)}, gt={len(gt_detections)}"
    )

    save_detected_holds(args.image, corners, detections_full, args.save_holds)
    plot_comparison(image_rgb, corners, detections_full, gt_detections, metrics_full, output_path=args.save_plot)


if __name__ == "__main__":
    main()
