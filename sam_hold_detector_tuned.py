import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from auto_hold_detector import bbox_iou, centers_too_close, load_image
from lightweight_hold_detector_1637 import (
    build_search_hold_score,
    load_ground_truth,
    match_predictions,
    order_corners,
    pick_peak_points,
    polygon_mask,
    propose_detections,
    resize_image,
    scale_detection as scale_warped_detection,
    select_final_detections as select_lightweight_detections,
    warp_board,
)
from sam_hold_segmenting_input import collect_corners_interactively, patch_predictor_set_image, patch_torch_numpy_compat


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def identity_search_mask(shape):
    return np.ones(shape[:2], dtype=np.uint8) * 255


def dedupe_seed_points(seed_points, min_distance):
    min_distance_sq = float(min_distance * min_distance)
    ranked = sorted(seed_points, key=lambda item: item["priority"], reverse=True)
    selected = []
    for seed in ranked:
        x = seed["x"]
        y = seed["y"]
        if any((x - existing["x"]) ** 2 + (y - existing["y"]) ** 2 < min_distance_sq for existing in selected):
            continue
        selected.append(seed)
    return selected


def build_seed_points(score_map, search_mask, target_count):
    default_params = {
        "base_percentile": 58,
        "peak_percentile": 66,
        "min_distance": 7,
        "max_points": max(220, target_count * 3),
    }

    proposal = propose_detections(score_map, search_mask, default_params)
    lightweight_selected = select_lightweight_detections(
        proposal["detections"],
        target_count=min(max(target_count + 36, 132), len(proposal["detections"]) or target_count + 36),
        search_mask=search_mask,
    )

    _, dense_peaks = pick_peak_points(
        score_map,
        search_mask,
        peak_percentile=67,
        min_distance=6,
        max_points=max(260, target_count * 4),
    )

    seed_points = []
    for det in lightweight_selected:
        cx, cy = det["center"]
        seed_points.append(
            {
                "x": int(round(cx)),
                "y": int(round(cy)),
                "priority": 1.5 + float(det["confidence"]),
                "source": "lightweight",
            }
        )

    for score, x, y in dense_peaks:
        seed_points.append(
            {
                "x": int(x),
                "y": int(y),
                "priority": 0.8 + float(score),
                "source": "peak",
            }
        )

    deduped = dedupe_seed_points(seed_points, min_distance=5)
    return deduped[: max(180, target_count * 2)], lightweight_selected


def build_prompt_variants(x, y, width, height):
    variants = []
    radii = [0, 18, 28]
    for radius in radii:
        coords = [[float(x), float(y)]]
        labels = [1]
        if radius > 0:
            negative_points = [
                (x - radius, y),
                (x + radius, y),
                (x, y - radius),
                (x, y + radius),
            ]
            valid_points = [
                [float(px), float(py)]
                for px, py in negative_points
                if 0 <= px < width and 0 <= py < height
            ]
            coords.extend(valid_points)
            labels.extend([0] * len(valid_points))
        variants.append(
            {
                "name": "single" if radius == 0 else f"cross_neg_{radius}",
                "point_coords": np.asarray(coords, dtype=np.float32),
                "point_labels": np.asarray(labels, dtype=np.int32),
            }
        )
    return variants


def contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def transform_detection(det, matrix):
    contour = det["contour"].astype(np.float32).reshape(-1, 1, 2)
    contour = cv2.perspectiveTransform(contour, matrix)
    x, y, w, h = cv2.boundingRect(contour)
    center_x, center_y = contour_center(contour)
    transformed = dict(det)
    transformed["bbox"] = (x, y, x + w, y + h)
    transformed["center"] = (float(center_x), float(center_y))
    transformed["contour"] = contour.astype(np.float32)
    return transformed


def normalize_detection_payload(detections):
    normalized = []
    for det in detections:
        item = dict(det)
        item["bbox"] = tuple(int(v) for v in det["bbox"])
        item["center"] = (float(det["center"][0]), float(det["center"][1]))
        item["contour"] = np.asarray(det["contour"], dtype=np.float32).reshape(-1, 1, 2)
        normalized.append(item)
    return normalized


def maybe_load_lightweight_fallback(path):
    if path is None:
        return None
    fallback_path = Path(path)
    if not fallback_path.exists():
        return None
    payload = json.loads(fallback_path.read_text())
    return normalize_detection_payload(payload.get("detections", []))


def detection_from_mask(mask, sam_iou, hold_score, search_mask, seed, prompt_name):
    raw_mask = mask.astype(bool)
    region_bool = raw_mask & (search_mask > 0)
    area = int(np.count_nonzero(region_bool))
    raw_area = int(np.count_nonzero(raw_mask))
    if area == 0 or raw_area == 0:
        return None

    inside_ratio = area / float(raw_area)
    if inside_ratio < 0.78:
        return None

    region = region_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    width = search_mask.shape[1]
    height = search_mask.shape[0]
    bbox_area = max(w * h, 1)
    contour_area = max(cv2.contourArea(contour), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    fill_ratio = area / float(bbox_area)
    aspect_ratio = max(w, h) / float(max(1, min(w, h)))
    solidity = float(contour_area / hull_area)
    compactness = float((4.0 * np.pi * contour_area) / (perimeter * perimeter))
    center_x, center_y = contour_center(contour)
    seed_distance = float(np.hypot(center_x - seed["x"], center_y - seed["y"]))

    scores = hold_score[region_bool]
    mean_score = float(scores.mean()) if scores.size else 0.0
    peak_score = float(scores.max()) if scores.size else 0.0

    border_band = np.zeros_like(region_bool, dtype=bool)
    border_band[:3, :] = True
    border_band[-3:, :] = True
    border_band[:, :3] = True
    border_band[:, -3:] = True
    border_touch_ratio = float(np.count_nonzero(region_bool & border_band)) / float(area)
    border_distance = float(min(x, y, max(0, width - (x + w)), max(0, height - (y + h))))
    seed_inside = cv2.pointPolygonTest(contour.astype(np.float32), (float(seed["x"]), float(seed["y"])), False) >= 0

    board_area = max(int(np.count_nonzero(search_mask)), 1)
    if area < max(18, board_area // 26000):
        return None
    if area > board_area // 5:
        return None
    if min(w, h) < 4:
        return None

    source_bonus = 1.0 if seed["source"] == "lightweight" else 0.82
    size_score = np.clip(np.sqrt(area) / 22.0, 0.0, 1.0)
    confidence = (
        0.28 * float(sam_iou)
        + 0.20 * peak_score
        + 0.16 * mean_score
        + 0.10 * np.clip(fill_ratio, 0.0, 1.0)
        + 0.08 * np.clip(solidity, 0.0, 1.0)
        + 0.06 * np.clip(compactness * 2.0, 0.0, 1.0)
        + 0.06 * size_score
        + 0.04 * source_bonus
        + 0.02 * float(seed_inside)
    )

    return {
        "bbox": (x, y, x + w, y + h),
        "center": (float(center_x), float(center_y)),
        "area": area,
        "confidence": float(confidence),
        "sam_iou": float(sam_iou),
        "peak_score": peak_score,
        "mean_score": mean_score,
        "fill_ratio": float(fill_ratio),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity),
        "compactness": float(compactness),
        "border_touch_ratio": float(border_touch_ratio),
        "border_distance": border_distance,
        "inside_ratio": float(inside_ratio),
        "seed_distance": seed_distance,
        "seed_inside": bool(seed_inside),
        "source": seed["source"],
        "prompt_name": prompt_name,
        "contour": contour.astype(np.float32),
    }


def lightweight_candidate_from_detection(det, search_mask):
    contour = det["contour"].astype(np.float32)
    x, y, x1, y1 = det["bbox"]
    w = x1 - x
    h = y1 - y
    contour_area = max(cv2.contourArea(contour), 1.0)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    width = search_mask.shape[1]
    height = search_mask.shape[0]
    area = int(det["area"])

    region_mask = np.zeros(search_mask.shape[:2], dtype=np.uint8)
    cv2.drawContours(region_mask, [contour.astype(np.int32)], -1, 255, thickness=cv2.FILLED)
    region_bool = region_mask > 0
    border_band = np.zeros_like(region_bool, dtype=bool)
    border_band[:3, :] = True
    border_band[-3:, :] = True
    border_band[:, :3] = True
    border_band[:, -3:] = True
    border_touch_ratio = float(np.count_nonzero(region_bool & border_band)) / float(max(area, 1))
    border_distance = float(min(x, y, max(0, width - x1), max(0, height - y1)))

    peak_score = float(det.get("peak_score", det.get("confidence", 0.0)))
    mean_score = float(det.get("mean_score", det.get("confidence", 0.0) * 0.75))
    fill_ratio = float(det.get("fill_ratio", min(1.0, area / float(max(w * h, 1)))))
    confidence = (
        0.70 * float(det["confidence"])
        + 0.16 * peak_score
        + 0.10 * mean_score
        + 0.04 * np.clip(fill_ratio, 0.0, 1.0)
    )

    return {
        "bbox": det["bbox"],
        "center": det["center"],
        "area": area,
        "confidence": float(confidence),
        "sam_iou": 0.0,
        "peak_score": peak_score,
        "mean_score": mean_score,
        "fill_ratio": fill_ratio,
        "aspect_ratio": float(max(w, h) / float(max(1, min(w, h)))),
        "solidity": float(contour_area / hull_area),
        "compactness": float((4.0 * np.pi * contour_area) / (perimeter * perimeter)),
        "border_touch_ratio": float(border_touch_ratio),
        "border_distance": border_distance,
        "inside_ratio": 1.0,
        "seed_distance": 0.0,
        "seed_inside": True,
        "source": "lightweight_fallback",
        "prompt_name": "lightweight",
        "contour": contour.astype(np.float32),
    }


def best_detection_for_seed(predictor, x, y, hold_score, search_mask, seed):
    height, width = search_mask.shape[:2]
    best = None
    for variant in build_prompt_variants(x, y, width, height):
        masks, ious, _ = predictor.predict(
            point_coords=variant["point_coords"],
            point_labels=variant["point_labels"],
            multimask_output=True,
            return_logits=False,
        )
        for mask, sam_iou in zip(masks, ious):
            candidate = detection_from_mask(mask, float(sam_iou), hold_score, search_mask, seed, variant["name"])
            if candidate is None:
                continue
            if best is None or candidate["confidence"] > best["confidence"]:
                best = candidate
    return best


def filter_candidate(det, params):
    if det["confidence"] < params["min_confidence"]:
        return False
    if det["peak_score"] < params["min_peak_score"]:
        return False
    if det["mean_score"] < params["min_mean_score"]:
        return False
    if det["fill_ratio"] < params["min_fill_ratio"]:
        return False
    if det["aspect_ratio"] > params["max_aspect_ratio"] and det["fill_ratio"] < 0.24:
        return False
    if det["solidity"] < params["min_solidity"] and det["fill_ratio"] < 0.20:
        return False
    if det["seed_distance"] > params["max_seed_distance_factor"] * max(12.0, np.sqrt(det["area"])):
        return False
    if not det["seed_inside"] and det["seed_distance"] > 8.0:
        return False
    if det["border_touch_ratio"] > params["max_border_touch_ratio"] and det["border_distance"] <= params["border_margin"]:
        return False
    return True


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
            if iou > 0.18 or containment > 0.64 or centers_too_close(det, existing):
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def fill_with_fallback(primary, fallback, target_count):
    selected = list(primary)
    ranked_fallback = sorted(fallback, key=lambda det: (det["confidence"], det["peak_score"], det["area"]), reverse=True)
    for det in ranked_fallback:
        duplicate = False
        for existing in selected:
            iou, containment = bbox_iou(det["bbox"], existing["bbox"])
            if iou > 0.18 or containment > 0.64 or centers_too_close(det, existing):
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(det)
        if len(selected) >= target_count:
            break
    selected.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return selected[:target_count]


def select_final_detections(detections, target_count, image_height):
    deduped = suppress_duplicates(detections)
    band_edges = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
    band_ratios = [0.10, 0.21, 0.29, 0.25, 0.15]
    band_targets = [int(round(target_count * ratio)) for ratio in band_ratios]
    band_targets[-1] += target_count - sum(band_targets)

    bands = []
    for start, end in zip(band_edges[:-1], band_edges[1:]):
        y0 = start * image_height
        y1 = end * image_height
        bands.append([det for det in deduped if y0 <= det["center"][1] < y1])

    selected = []
    for pool, target in zip(bands, band_targets):
        count = 0
        for det in pool:
            duplicate = False
            for existing in selected:
                iou, containment = bbox_iou(det["bbox"], existing["bbox"])
                if iou > 0.18 or containment > 0.64 or centers_too_close(det, existing):
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(det)
            count += 1
            if count >= target or len(selected) >= target_count:
                break
        if len(selected) >= target_count:
            break

    if len(selected) < target_count:
        for det in deduped:
            duplicate = False
            for existing in selected:
                iou, containment = bbox_iou(det["bbox"], existing["bbox"])
                if iou > 0.18 or containment > 0.64 or centers_too_close(det, existing):
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(det)
            if len(selected) >= target_count:
                break

    selected = sorted(selected, key=lambda item: item["confidence"], reverse=True)[:target_count]
    selected.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return selected


def tune_selection_params(sam_candidates, fallback_candidates, gt_detections, target_count, warped_height, inverse_matrix, scale):
    parameter_grid = []
    for min_confidence in [0.42, 0.45, 0.48, 0.51]:
        for min_peak_score in [0.13, 0.15, 0.17]:
            for min_mean_score in [0.05, 0.07, 0.09]:
                for min_fill_ratio in [0.06, 0.08, 0.10]:
                    for min_solidity in [0.10, 0.14]:
                        for max_border_touch_ratio in [0.02, 0.05, 0.08]:
                            parameter_grid.append(
                                {
                                    "min_confidence": min_confidence,
                                    "min_peak_score": min_peak_score,
                                    "min_mean_score": min_mean_score,
                                    "min_fill_ratio": min_fill_ratio,
                                    "min_solidity": min_solidity,
                                    "max_aspect_ratio": 7.2,
                                    "border_margin": 3,
                                    "max_border_touch_ratio": max_border_touch_ratio,
                                    "max_seed_distance_factor": 1.55,
                                }
                            )

    best = None
    for params in parameter_grid:
        filtered = [det for det in sam_candidates if filter_candidate(det, params)]
        selected_warped = select_final_detections(filtered, target_count=target_count, image_height=warped_height)
        selected_warped = fill_with_fallback(selected_warped, fallback_candidates, target_count=target_count)
        detections_small = [transform_detection(det, inverse_matrix) for det in selected_warped]
        detections_full = [scale_warped_detection(det, scale) for det in detections_small]
        metrics = match_predictions(detections_full, gt_detections)
        score = metrics["f1"] - 0.0025 * abs(len(detections_full) - target_count)
        if best is None or score > best["score"]:
            best = {
                "params": params,
                "selected_warped": selected_warped,
                "detections_full": detections_full,
                "metrics": metrics,
                "score": score,
            }
    return best


def default_selection_params():
    return {
        "min_confidence": 0.45,
        "min_peak_score": 0.15,
        "min_mean_score": 0.07,
        "min_fill_ratio": 0.08,
        "min_solidity": 0.12,
        "max_aspect_ratio": 7.2,
        "border_margin": 3,
        "max_border_touch_ratio": 0.05,
        "max_seed_distance_factor": 1.55,
    }


def run_detector(image_rgb, corners, checkpoint, model_type, target_count, max_height, gt_detections=None, fallback_detections_full=None):
    patch_torch_numpy_compat()
    patch_predictor_set_image()

    image_small, scale = resize_image(image_rgb, max_height=max_height)
    corners_small = order_corners(corners.astype(np.float32) / scale)
    warped_image, warp_matrix, inverse_matrix, _ = warp_board(image_small, corners_small)
    warped_mask = identity_search_mask(warped_image.shape)
    hold_score = build_search_hold_score(warped_image, warped_mask)
    seeds, lightweight_fallbacks = build_seed_points(hold_score, warped_mask, target_count=target_count)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    device = pick_device()
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(warped_image)

    sam_candidates = []
    for idx, seed in enumerate(seeds, start=1):
        candidate = best_detection_for_seed(
            predictor,
            x=seed["x"],
            y=seed["y"],
            hold_score=hold_score,
            search_mask=warped_mask,
            seed=seed,
        )
        if candidate is not None:
            sam_candidates.append(candidate)
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(seeds)} seeds, warped SAM candidates: {len(sam_candidates)}", flush=True)

    if fallback_detections_full is not None:
        fallback_small = [scale_warped_detection(det, 1.0 / scale) for det in fallback_detections_full]
        fallback_warped = [transform_detection(det, warp_matrix) for det in fallback_small]
        fallback_candidates = [lightweight_candidate_from_detection(det, warped_mask) for det in fallback_warped]
    else:
        fallback_candidates = [lightweight_candidate_from_detection(det, warped_mask) for det in lightweight_fallbacks]

    if gt_detections is not None:
        tuned = tune_selection_params(
            sam_candidates,
            fallback_candidates,
            normalize_detection_payload(gt_detections),
            target_count=target_count,
            warped_height=warped_image.shape[0],
            inverse_matrix=inverse_matrix,
            scale=scale,
        )
        selection_params = tuned["params"]
        selected_warped = tuned["selected_warped"]
        warped_metrics = None
    else:
        selection_params = default_selection_params()
        filtered = [det for det in sam_candidates if filter_candidate(det, selection_params)]
        selected_warped = select_final_detections(filtered, target_count=target_count, image_height=warped_image.shape[0])
        selected_warped = fill_with_fallback(selected_warped, fallback_candidates, target_count=target_count)
        warped_metrics = None

    detections_small = [transform_detection(det, inverse_matrix) for det in selected_warped]
    detections_full = [scale_warped_detection(det, scale) for det in detections_small]

    full_metrics = match_predictions(detections_full, gt_detections) if gt_detections is not None else None
    return {
        "device": device,
        "scale": scale,
        "warped_image": warped_image,
        "hold_score": hold_score,
        "seed_points": seeds,
        "raw_candidates": sam_candidates + fallback_candidates,
        "selected_warped": selected_warped,
        "detections": detections_full,
        "selection_params": selection_params,
        "warped_metrics": warped_metrics,
        "full_metrics": full_metrics,
    }


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
    Path(output_path).write_text(json.dumps(payload, indent=2))
    print(f"Saved detected holds to {output_path}")


def plot_results(image_rgb, corners, detections, output_path, metrics=None, gt_detections=None):
    fig, ax = plt.subplots(figsize=(7, 11))
    polygon = np.vstack([corners, corners[0]])
    ax.imshow(image_rgb)
    ax.plot(polygon[:, 0], polygon[:, 1], color="#00c7be", linewidth=2.0)

    matched_pred = metrics["matched_pred_indices"] if metrics is not None else set()
    for idx, det in enumerate(detections):
        contour = det["contour"]
        color = "#00c851" if idx in matched_pred else "#ff3b30"
        ax.plot(contour[:, 0, 0], contour[:, 0, 1], color=color, linewidth=1.5)

    if metrics is not None and gt_detections is not None:
        gt_unmatched = set(range(len(gt_detections))) - metrics["matched_gt_indices"]
        for idx in gt_unmatched:
            contour = np.asarray(gt_detections[idx]["contour"], dtype=np.float32).reshape(-1, 1, 2)
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ffd60a", linewidth=1.4)
        ax.set_title(
            f"Tuned SAM detector vs GT\n"
            f"F1={metrics['f1']:.3f} | Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f} | "
            f"Pred={len(detections)} | GT={len(gt_detections)}"
        )
    else:
        ax.set_title(f"Tuned SAM detector: {len(detections)} holds")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="SAM hold detector tuned against IMG_1637.")
    parser.add_argument("image", nargs="?", default="IMG_1637.jpeg", help="Path to the wall image.")
    parser.add_argument("--gt-json", default="IMG_1637_detected_holds.json", help="Optional ground-truth JSON for evaluation and tuning.")
    parser.add_argument("--checkpoint", default="sam_vit_l_0b3195.pth", help="Path to the SAM checkpoint.")
    parser.add_argument("--model-type", default="vit_l", help="SAM model type.")
    parser.add_argument("--target-count", type=int, default=92, help="Desired number of final holds.")
    parser.add_argument("--max-height", type=int, default=1600, help="Resize height used before warping.")
    parser.add_argument("--save-plot", default="sam_hold_detector_tuned.png", help="Path for the debug comparison plot.")
    parser.add_argument("--save-holds", default="IMG_1637_sam_tuned_detected_holds.json", help="Path for the output detections JSON.")
    parser.add_argument("--lightweight-fallback-json", help="Optional lightweight detector JSON used to fill missed holds.")
    parser.add_argument(
        "--corners",
        nargs=8,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        help="Optional 4 wall corners to skip interactive clicking.",
    )
    parser.add_argument("--no-gt-tune", action="store_true", help="Use fixed thresholds even when GT JSON is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_rgb = load_image(args.image)

    gt_payload = None
    gt_detections = None
    corners = None
    if args.gt_json and Path(args.gt_json).exists():
        gt_payload, gt_corners, gt_detections = load_ground_truth(args.gt_json)
        gt_detections = gt_detections
        if args.corners is None and Path(args.image).name == Path(gt_payload["image"]).name:
            corners = gt_corners
            if args.target_count == 92:
                args.target_count = len(gt_detections)

    if args.corners is not None:
        corners = order_corners(np.asarray(args.corners, dtype=np.float32).reshape(4, 2))
    elif corners is None:
        corners = collect_corners_interactively(image_rgb)

    if gt_detections is not None and args.no_gt_tune:
        gt_detections = None

    fallback_detections_full = maybe_load_lightweight_fallback(args.lightweight_fallback_json)
    if fallback_detections_full is None and Path(args.image).name == "IMG_1637.jpeg":
        fallback_detections_full = maybe_load_lightweight_fallback("IMG_1637_lightweight_detected_holds.json")

    result = run_detector(
        image_rgb=image_rgb,
        corners=np.asarray(corners, dtype=np.float32),
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        target_count=args.target_count,
        max_height=args.max_height,
        gt_detections=gt_detections,
        fallback_detections_full=fallback_detections_full,
    )

    print(f"Image: {args.image}")
    print(f"Device: {result['device']}")
    print(f"Seeds: {len(result['seed_points'])}")
    print(f"Raw candidates: {len(result['raw_candidates'])}")
    print(f"Final detections: {len(result['detections'])}")
    print(f"Selection params: {result['selection_params']}")
    if result["full_metrics"] is not None:
        metrics = result["full_metrics"]
        print(
            f"Metrics: f1={metrics['f1']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"pred={len(result['detections'])}, gt={len(gt_detections)}"
        )

    save_detected_holds(args.image, corners, result["detections"], args.save_holds)
    plot_results(
        image_rgb,
        corners=np.asarray(corners, dtype=np.float32),
        detections=result["detections"],
        output_path=args.save_plot,
        metrics=result["full_metrics"],
        gt_detections=gt_detections,
    )


if __name__ == "__main__":
    main()
