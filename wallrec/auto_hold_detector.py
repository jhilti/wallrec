import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


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


def normalize_map(values, mask=None):
    valid = values[mask] if mask is not None else values.reshape(-1)
    if valid.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(valid, [5, 95])
    span = max(float(hi - lo), 1e-6)
    normalized = np.clip((values - lo) / span, 0.0, 1.0)
    return normalized.astype(np.float32)


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def extend_mask_downward(mask, image_small):
    height, width = mask.shape
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask

    x_left = int(np.percentile(xs, 4))
    x_right = int(np.percentile(xs, 96))
    y_bottom = int(np.percentile(ys, 99))

    gray = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.convertScaleAbs(np.abs(grad_x) + 0.6 * np.abs(grad_y))
    edge_mask = (edge_strength > np.percentile(edge_strength, 70)).astype(np.uint8) * 255

    seed = np.zeros_like(mask)
    extend_top = max(0, y_bottom - height // 20)
    extend_left = max(0, x_left - width // 50)
    extend_right = min(width, x_right + width // 50)
    seed[extend_top:, extend_left:extend_right] = 255

    extended = cv2.bitwise_or(mask, cv2.bitwise_and(seed, cv2.bitwise_not(edge_mask)))
    extended = cv2.morphologyEx(
        extended,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=2,
    )
    extended = cv2.morphologyEx(
        extended,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    return keep_largest_component(extended)


def detect_wall_mask(image_small):
    rgb = image_small.astype(np.float32) / 255.0
    lab = cv2.cvtColor(image_small, cv2.COLOR_RGB2LAB).astype(np.float32)

    height, width = image_small.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx / max(width - 1, 1)
    y_norm = yy / max(height - 1, 1)

    brightness = rgb.mean(axis=-1)
    warm_score = 1.35 * rgb[:, :, 0] + 0.85 * rgb[:, :, 1] - 1.45 * rgb[:, :, 2]
    lab_yellow = np.clip((lab[:, :, 2] - 128.0) / 45.0, 0.0, 1.0)
    center_prior = np.clip(1.0 - np.abs(x_norm - 0.53) / 0.62, 0.0, 1.0)
    bottom_support = 0.86 + 0.14 * np.clip((y_norm - 0.55) / 0.45, 0.0, 1.0)

    wall_score = (
        0.52 * normalize_map(warm_score)
        + 0.28 * lab_yellow
        + 0.20 * normalize_map(brightness)
    )
    wall_score *= (0.58 + 0.42 * center_prior) * bottom_support

    wall_mask = (wall_score > np.percentile(wall_score, 58)).astype(np.uint8) * 255
    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=2,
    )
    wall_mask = cv2.morphologyEx(
        wall_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        iterations=1,
    )
    wall_mask = keep_largest_component(wall_mask)

    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        refined = np.zeros_like(wall_mask)
        cv2.drawContours(refined, [hull], -1, 255, thickness=cv2.FILLED)
        wall_mask = cv2.bitwise_and(refined, wall_mask)
        wall_mask = cv2.morphologyEx(
            wall_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
            iterations=1,
        )

    wall_mask = extend_mask_downward(wall_mask, image_small)

    return wall_mask, wall_score.astype(np.float32)


def build_hold_score(image_small, wall_mask):
    rgb = image_small.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image_small, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(image_small, cv2.COLOR_RGB2LAB).astype(np.float32)

    rgb_blur = cv2.GaussianBlur(rgb, (0, 0), 17)
    lab_blur = cv2.GaussianBlur(lab, (0, 0), 19)

    color_delta = np.linalg.norm(rgb - rgb_blur, axis=-1)
    lab_delta = np.linalg.norm(lab - lab_blur, axis=-1)
    saturation = hsv[:, :, 1] / 255.0
    edges = cv2.Canny(image_small, 60, 150).astype(np.float32) / 255.0
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = np.abs(lap)

    wall_bool = wall_mask > 0
    color_delta = normalize_map(color_delta, wall_bool)
    lab_delta = normalize_map(lab_delta, wall_bool)
    saturation = normalize_map(saturation, wall_bool)
    edges = normalize_map(edges, wall_bool)
    texture = normalize_map(texture, wall_bool)

    score = (
        0.36 * lab_delta
        + 0.24 * color_delta
        + 0.16 * saturation
        + 0.14 * texture
        + 0.10 * edges
    )
    score *= wall_bool
    return score.astype(np.float32)


def pick_peak_points(score_map, wall_mask, peak_percentile, min_distance):
    wall_bool = wall_mask > 0
    threshold = max(0.18, np.percentile(score_map[wall_bool], peak_percentile))
    smoothed = cv2.GaussianBlur(score_map, (0, 0), 2.2)
    kernel_size = 2 * min_distance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(smoothed, kernel)
    peak_mask = (smoothed >= threshold) & wall_bool & (smoothed >= dilated - 1e-6)

    ys, xs = np.where(peak_mask)
    candidates = sorted(
        [(float(smoothed[y, x]), int(x), int(y)) for y, x in zip(ys, xs)],
        reverse=True,
    )

    peaks = []
    min_distance_sq = float(min_distance * min_distance)
    for score, x, y in candidates:
        keep = True
        for _, px, py in peaks:
            if (x - px) ** 2 + (y - py) ** 2 < min_distance_sq:
                keep = False
                break
        if keep:
            peaks.append((score, x, y))

    return smoothed, peaks


def split_component_by_peaks(component_mask, peaks_in_component):
    ys, xs = np.where(component_mask)
    if len(peaks_in_component) == 1:
        return [component_mask]

    coords = np.column_stack((xs, ys)).astype(np.float32)
    peak_coords = np.array([[peak[1], peak[2]] for peak in peaks_in_component], dtype=np.float32)
    distances = ((coords[:, None, :] - peak_coords[None, :, :]) ** 2).sum(axis=2)
    assignments = np.argmin(distances, axis=1)

    regions = []
    for peak_index in range(len(peaks_in_component)):
        assigned = assignments == peak_index
        if not np.any(assigned):
            continue
        region = np.zeros_like(component_mask, dtype=np.uint8)
        region[ys[assigned], xs[assigned]] = 255
        regions.append(region)
    return regions


def extract_regions_from_peaks(score_map, wall_mask, peaks, base_percentile):
    wall_bool = wall_mask > 0
    threshold = max(0.12, np.percentile(score_map[wall_bool], base_percentile))
    base_mask = (score_map >= threshold).astype(np.uint8) * 255
    base_mask = cv2.bitwise_and(base_mask, wall_mask)
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
    peak_regions = []
    for label in range(1, num_labels):
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        peaks_in_component = [peak for peak in peaks if labels[peak[2], peak[1]] == label]
        if not peaks_in_component:
            continue
        peak_regions.extend(split_component_by_peaks(component_mask, peaks_in_component))
    return base_mask, peak_regions


def detections_from_regions(peak_regions, score_map, wall_mask):
    wall_area = max(int(np.count_nonzero(wall_mask)), 1)
    min_area = max(16, wall_area // 6000)
    max_area = wall_area // 7

    detections = []
    cleaned_mask = np.zeros_like(wall_mask)
    for region in peak_regions:
        area = cv2.countNonZero(region)
        if area < min_area or area > max_area:
            continue

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = max(w * h, 1)
        fill_ratio = area / float(bbox_area)
        aspect_ratio = max(w, h) / float(max(1, min(w, h)))
        contour_area = max(cv2.contourArea(contour), 1.0)
        perimeter = max(cv2.arcLength(contour, True), 1.0)
        hull = cv2.convexHull(contour)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = float(contour_area / hull_area)
        compactness = float((4.0 * np.pi * contour_area) / (perimeter * perimeter))
        region_bool = region > 0
        region_scores = score_map[region_bool]
        mean_score = float(region_scores.mean()) if region_scores.size else 0.0
        peak_score = float(region_scores.max()) if region_scores.size else 0.0

        if min(w, h) < 4:
            continue
        if fill_ratio < 0.08:
            continue
        if aspect_ratio > 7.0 and fill_ratio < 0.30:
            continue
        if solidity < 0.22 and fill_ratio < 0.22:
            continue
        if mean_score < 0.16 or peak_score < 0.22:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            center_x = x + w / 2.0
            center_y = y + h / 2.0
        else:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]

        size_score = np.clip(np.sqrt(area) / 24.0, 0.0, 1.0)
        confidence = (
            0.44 * peak_score
            + 0.24 * mean_score
            + 0.10 * np.clip(fill_ratio, 0.0, 1.0)
            + 0.08 * np.clip(solidity, 0.0, 1.0)
            + 0.06 * np.clip(compactness * 2.0, 0.0, 1.0)
            + 0.08 * size_score
        )

        detections.append(
            {
                "bbox": (x, y, x + w, y + h),
                "area": area,
                "center": (center_x, center_y),
                "fill_ratio": fill_ratio,
                "mean_score": mean_score,
                "peak_score": peak_score,
                "confidence": float(confidence),
                "solidity": solidity,
                "compactness": compactness,
                "region_mask": region.copy(),
                "contour": contour.copy(),
            }
        )

    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return cleaned_mask, detections


def bbox_iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0, 0.0
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1)
    area_b = max((bx1 - bx0) * (by1 - by0), 1)
    union = area_a + area_b - intersection
    return intersection / float(union), intersection / float(min(area_a, area_b))


def centers_too_close(det_a, det_b):
    ax0, ay0, ax1, ay1 = det_a["bbox"]
    bx0, by0, bx1, by1 = det_b["bbox"]
    aw = ax1 - ax0
    ah = ay1 - ay0
    bw = bx1 - bx0
    bh = by1 - by0
    ax, ay = det_a["center"]
    bx, by = det_b["center"]
    center_distance = np.hypot(ax - bx, ay - by)
    size_limit = 0.38 * min(max(aw, ah), max(bw, bh))
    return center_distance < size_limit


def suppress_duplicates(ranked):
    kept = []
    for detection in ranked:
        duplicate = False
        for existing in kept:
            iou, containment = bbox_iou(detection["bbox"], existing["bbox"])
            if iou > 0.18 or containment > 0.62 or centers_too_close(detection, existing):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def consolidate_detections(detections, target_count, image_height):
    ranked = sorted(
        detections,
        key=lambda det: (det["confidence"], det["peak_score"], det["mean_score"], det["area"]),
        reverse=True,
    )
    deduped = suppress_duplicates(ranked)

    lower_band_start = 0.78 * image_height
    kickboard_start = 0.90 * image_height

    kickboard = [det for det in deduped if det["center"][1] >= kickboard_start]
    lower = [det for det in deduped if det["center"][1] >= lower_band_start and det["center"][1] < kickboard_start]
    upper = [det for det in deduped if det["center"][1] < lower_band_start]

    desired_kickboard = max(4, target_count // 18)
    desired_lower = max(20, target_count // 4)

    selected = []
    for pool, limit in [
        (kickboard, desired_kickboard),
        (lower, desired_lower),
        (upper, target_count),
        (lower, target_count),
        (kickboard, target_count),
    ]:
        for det in pool:
            if len([x for x in selected if x["center"][1] >= kickboard_start]) >= desired_kickboard and pool is kickboard:
                break
            if len([x for x in selected if lower_band_start <= x["center"][1] < kickboard_start]) >= desired_lower and pool is lower:
                break
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


def tune_holds(hold_score, wall_mask, target_count=92):
    best = None
    parameter_grid = [
        (64, 72, 8),
        (64, 74, 9),
        (66, 74, 8),
        (66, 76, 9),
        (68, 76, 10),
        (68, 78, 10),
        (70, 78, 11),
        (70, 80, 11),
    ]

    for base_percentile, peak_percentile, min_distance in parameter_grid:
        smoothed_score, peaks = pick_peak_points(
            hold_score,
            wall_mask,
            peak_percentile=peak_percentile,
            min_distance=min_distance,
        )
        base_mask, peak_regions = extract_regions_from_peaks(
            smoothed_score,
            wall_mask,
            peaks,
            base_percentile=base_percentile,
        )
        _, raw_detections = detections_from_regions(peak_regions, smoothed_score, wall_mask)
        detections = consolidate_detections(
            raw_detections,
            target_count=target_count,
            image_height=wall_mask.shape[0],
        )
        cleaned_mask = np.zeros_like(wall_mask)
        for detection in detections:
            cleaned_mask = cv2.bitwise_or(cleaned_mask, detection["region_mask"])

        count = len(detections)
        count_penalty = abs(count - target_count)
        over_penalty = max(0, count - target_count) * 0.35
        score = count_penalty + over_penalty

        candidate = {
            "strong_mask": np.where(smoothed_score >= np.percentile(smoothed_score[wall_mask > 0], peak_percentile), 255, 0).astype(np.uint8),
            "mid_mask": base_mask,
            "cleaned_mask": cleaned_mask,
            "detections": detections,
            "peaks": peaks,
            "smoothed_score": smoothed_score,
            "params": (base_percentile, peak_percentile, min_distance),
            "score": score,
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    return best


def scale_detection(detection, scale):
    x0, y0, x1, y1 = detection["bbox"]
    center_x, center_y = detection["center"]
    contour = detection["contour"].astype(np.float32).copy()
    contour[:, 0, 0] *= scale
    contour[:, 0, 1] *= scale
    return {
        "bbox": (
            int(round(x0 * scale)),
            int(round(y0 * scale)),
            int(round(x1 * scale)),
            int(round(y1 * scale)),
        ),
        "area": float(detection["area"] * scale * scale),
        "center": (float(center_x * scale), float(center_y * scale)),
        "fill_ratio": detection["fill_ratio"],
        "confidence": detection["confidence"],
        "contour": contour,
    }


def detect_holds(image_rgb, target_count=92):
    image_small, scale = resize_for_detection(image_rgb, max_height=1400)
    wall_mask, wall_score = detect_wall_mask(image_small)
    hold_score = build_hold_score(image_small, wall_mask)
    tuned = tune_holds(hold_score, wall_mask, target_count=target_count)
    detections_small = tuned["detections"]
    detections = [scale_detection(detection, scale) for detection in detections_small]

    return {
        "image_small": image_small,
        "scale": scale,
        "wall_mask": wall_mask,
        "wall_score": wall_score,
        "hold_score": hold_score,
        "strong_mask": tuned["strong_mask"],
        "mid_mask": tuned["mid_mask"],
        "cleaned_mask": tuned["cleaned_mask"],
        "peaks": tuned["peaks"],
        "params": tuned["params"],
        "detections": detections,
    }


def plot_results(image_rgb, result, image_path, save_path=None):
    detections = result["detections"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    ax_overlay, ax_wall, ax_score, ax_mask = axes.ravel()

    ax_overlay.imshow(image_rgb)
    for idx, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = detection["bbox"]
        contour = detection["contour"]
        ax_overlay.plot(
            contour[:, 0, 0],
            contour[:, 0, 1],
            color="#ff3b30",
            linewidth=1.6,
        )
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=1.0,
            edgecolor="#ff3b30",
            facecolor="none",
            alpha=0.45,
        )
        ax_overlay.add_patch(rect)
        ax_overlay.text(
            x0,
            max(y0 - 8, 10),
            str(idx),
            color="white",
            fontsize=10,
            bbox={"facecolor": "#ff3b30", "edgecolor": "none", "pad": 1.2},
        )
    ax_overlay.set_title(f"Automatic hold detections: {len(detections)}")
    ax_overlay.axis("off")

    ax_wall.imshow(result["image_small"])
    ax_wall.imshow(result["wall_mask"], alpha=0.32, cmap="Greens")
    ax_wall.set_title("Detected wall mask")
    ax_wall.axis("off")

    score_view = ax_score.imshow(result["hold_score"], cmap="magma")
    ax_score.set_title("Hold score heatmap")
    ax_score.axis("off")
    fig.colorbar(score_view, ax=ax_score, fraction=0.046, pad=0.04)

    ax_mask.imshow(result["image_small"])
    ax_mask.imshow(result["cleaned_mask"], alpha=0.45, cmap="cool")
    ax_mask.set_title("Filtered hold mask")
    ax_mask.axis("off")

    fig.suptitle(image_path, fontsize=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Automatic climbing hold detector draft using OpenCV.")
    parser.add_argument("image", nargs="?", default="IMG_1505.jpeg", help="Path to the wall image.")
    parser.add_argument("--save", help="Optional path to save the debug plot.")
    parser.add_argument("--target-count", type=int, default=92, help="Expected number of holds to bias tuning.")
    args = parser.parse_args()

    image_rgb = load_image(args.image)
    result = detect_holds(image_rgb, target_count=args.target_count)

    print(f"Image: {args.image}")
    print(f"Tuned params: base_pct={result['params'][0]}, peak_pct={result['params'][1]}, min_distance={result['params'][2]}")
    print(f"Detections: {len(result['detections'])}")
    for idx, detection in enumerate(result["detections"], start=1):
        x0, y0, x1, y1 = detection["bbox"]
        center_x, center_y = detection["center"]
        print(
            f"{idx:02d}: bbox=({x0}, {y0}) -> ({x1}, {y1}), "
            f"center=({center_x:.1f}, {center_y:.1f}), area~{detection['area']:.0f}px"
        )

    plot_results(image_rgb, result, args.image, save_path=args.save)


if __name__ == "__main__":
    main()
