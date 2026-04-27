import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything.utils.amg import MaskData

from auto_hold_detector import (
    bbox_iou,
    build_hold_score,
    centers_too_close,
    detect_wall_mask,
    load_image,
    resize_for_detection,
)

_ORIGINAL_TORCH_AS_TENSOR = torch.as_tensor
_ORIGINAL_TENSOR_ARRAY = torch.Tensor.__array__


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def patch_torch_as_tensor_for_numpy2():
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
            np_dtype = data.dtype

            if dtype is None:
                dtype = dtype_map.get(np_dtype, None)

            if np_dtype == np.uint8 and data.flags["C_CONTIGUOUS"]:
                tensor = torch.frombuffer(bytearray(data.tobytes()), dtype=torch.uint8).clone().reshape(data.shape)
            else:
                tensor = torch.tensor(data.tolist(), dtype=dtype)

            if device is not None:
                tensor = tensor.to(device)
            return tensor

        return _ORIGINAL_TORCH_AS_TENSOR(data, *args, **kwargs)

    torch.as_tensor = patched_as_tensor


def patch_torch_tensor_array_for_numpy2():
    def patched_array(self, dtype=None):
        array = np.asarray(self.detach().cpu().tolist())
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    torch.Tensor.__array__ = patched_array


def load_sam_model(model_type, checkpoint):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    device = pick_device()
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()
    return sam, device


def patch_sam_predictor_for_numpy2():
    def patched_set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        input_image = self.transform.apply_image(image)
        buffer = input_image.tobytes()
        input_image_torch = torch.frombuffer(buffer, dtype=torch.uint8).clone()
        input_image_torch = input_image_torch.reshape(input_image.shape).to(self.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        self.set_torch_image(input_image_torch, image.shape[:2])

    SamPredictor.set_image = patched_set_image


def patch_maskdata_for_numpy2():
    def safe_tensor_to_numpy(tensor):
        cpu = tensor.detach().cpu()
        return np.asarray(cpu.tolist())

    def patched_filter(self, keep):
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                keep_np = np.asarray(keep.detach().cpu().tolist()) if isinstance(keep, torch.Tensor) else np.asarray(keep)
                self._stats[k] = v[keep_np]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                keep_list = keep.detach().cpu().tolist()
                self._stats[k] = [a for i, a in enumerate(v) if keep_list[i]]
            elif isinstance(v, list):
                keep_list = keep.detach().cpu().tolist() if isinstance(keep, torch.Tensor) else keep
                self._stats[k] = [v[i] for i in keep_list]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def patched_to_numpy(self):
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = safe_tensor_to_numpy(v)

    MaskData.filter = patched_filter
    MaskData.to_numpy = patched_to_numpy


def build_mask_generator(
    sam,
    points_per_side=24,
    crop_n_layers=1,
    pred_iou_thresh=0.84,
    stability_score_thresh=0.90,
    min_mask_region_area=30,
):
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_region_area,
    )


def mask_to_detection(annotation, hold_score, wall_mask):
    region_bool = annotation["segmentation"]
    region_u8 = region_bool.astype(np.uint8) * 255
    area = int(annotation["area"])
    wall_overlap = float(np.count_nonzero(region_bool & (wall_mask > 0))) / max(area, 1)

    if wall_overlap < 0.70:
        return None

    contours, _ = cv2.findContours(region_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

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

    if area < 18:
        return None
    if min(w, h) < 4:
        return None
    if aspect_ratio > 7.0 and fill_ratio < 0.30:
        return None
    if solidity < 0.18 and fill_ratio < 0.20:
        return None

    region_scores = hold_score[region_bool]
    mean_score = float(region_scores.mean()) if region_scores.size else 0.0
    peak_score = float(region_scores.max()) if region_scores.size else 0.0

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
        0.32 * float(annotation.get("predicted_iou", 0.0))
        + 0.18 * float(annotation.get("stability_score", 0.0))
        + 0.18 * peak_score
        + 0.12 * mean_score
        + 0.08 * np.clip(fill_ratio, 0.0, 1.0)
        + 0.06 * np.clip(solidity, 0.0, 1.0)
        + 0.06 * np.clip(compactness * 2.0, 0.0, 1.0)
        + 0.08 * size_score
    )

    return {
        "bbox": (x, y, x + w, y + h),
        "area": area,
        "center": (center_x, center_y),
        "fill_ratio": fill_ratio,
        "mean_score": mean_score,
        "peak_score": peak_score,
        "confidence": float(confidence),
        "solidity": solidity,
        "compactness": compactness,
        "region_mask": region_u8,
        "contour": contour.copy(),
        "predicted_iou": float(annotation.get("predicted_iou", 0.0)),
        "stability_score": float(annotation.get("stability_score", 0.0)),
    }


def consolidate_sam_detections(detections, target_count, image_height):
    ranked = sorted(
        detections,
        key=lambda det: (
            det["confidence"],
            det["predicted_iou"],
            det["stability_score"],
            det["peak_score"],
            det["area"],
        ),
        reverse=True,
    )

    deduped = []
    for detection in ranked:
        duplicate = False
        for existing in deduped:
            iou, containment = bbox_iou(detection["bbox"], existing["bbox"])
            if iou > 0.20 or containment > 0.68 or centers_too_close(detection, existing):
                duplicate = True
                break
        if not duplicate:
            deduped.append(detection)

    lower_band_start = 0.78 * image_height
    kickboard_start = 0.90 * image_height
    desired_kickboard = max(4, target_count // 18)
    desired_lower = max(20, target_count // 4)

    kickboard = [det for det in deduped if det["center"][1] >= kickboard_start]
    lower = [det for det in deduped if lower_band_start <= det["center"][1] < kickboard_start]
    upper = [det for det in deduped if det["center"][1] < lower_band_start]

    selected = []
    for pool, target in [
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

            kick_count = sum(x["center"][1] >= kickboard_start for x in selected)
            lower_count = sum(lower_band_start <= x["center"][1] < kickboard_start for x in selected)
            if pool is kickboard and kick_count >= target:
                break
            if pool is lower and lower_count >= target:
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
        "confidence": detection["confidence"],
        "contour": contour,
    }


def detect_holds_with_sam(
    image_rgb,
    checkpoint,
    model_type="vit_l",
    target_count=92,
    max_height=1400,
    points_per_side=24,
    crop_n_layers=1,
):
    image_small, scale = resize_for_detection(image_rgb, max_height=max_height)
    wall_mask, wall_score = detect_wall_mask(image_small)
    hold_score = build_hold_score(image_small, wall_mask)

    patch_torch_as_tensor_for_numpy2()
    patch_torch_tensor_array_for_numpy2()
    patch_sam_predictor_for_numpy2()
    patch_maskdata_for_numpy2()
    sam, device = load_sam_model(model_type=model_type, checkpoint=checkpoint)
    mask_generator = build_mask_generator(
        sam,
        points_per_side=points_per_side,
        crop_n_layers=crop_n_layers,
    )
    annotations = mask_generator.generate(image_small)

    detections = []
    for annotation in annotations:
        det = mask_to_detection(annotation, hold_score, wall_mask)
        if det is not None:
            detections.append(det)

    detections = consolidate_sam_detections(
        detections,
        target_count=target_count,
        image_height=image_small.shape[0],
    )

    cleaned_mask = np.zeros_like(wall_mask)
    for detection in detections:
        cleaned_mask = cv2.bitwise_or(cleaned_mask, detection["region_mask"])

    scaled = [scale_detection(det, scale) for det in detections]
    return {
        "image_small": image_small,
        "scale": scale,
        "device": device,
        "wall_mask": wall_mask,
        "wall_score": wall_score,
        "hold_score": hold_score,
        "cleaned_mask": cleaned_mask,
        "raw_mask_count": len(annotations),
        "detections": scaled,
    }


def plot_results(image_rgb, result, image_path, save_path=None):
    detections = result["detections"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    ax_overlay, ax_wall, ax_score, ax_mask = axes.ravel()

    ax_overlay.imshow(image_rgb)
    for idx, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = detection["bbox"]
        contour = detection["contour"]
        ax_overlay.plot(contour[:, 0, 0], contour[:, 0, 1], color="#ff3b30", linewidth=1.5)
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
    ax_overlay.set_title(f"SAM hold detections: {len(detections)}")
    ax_overlay.axis("off")

    ax_wall.imshow(result["image_small"])
    ax_wall.imshow(result["wall_mask"], alpha=0.32, cmap="Greens")
    ax_wall.set_title("Detected wall mask")
    ax_wall.axis("off")

    score_view = ax_score.imshow(result["hold_score"], cmap="magma")
    ax_score.set_title("Shared hold score heatmap")
    ax_score.axis("off")
    fig.colorbar(score_view, ax=ax_score, fraction=0.046, pad=0.04)

    ax_mask.imshow(result["image_small"])
    ax_mask.imshow(result["cleaned_mask"], alpha=0.45, cmap="cool")
    ax_mask.set_title("Filtered SAM masks")
    ax_mask.axis("off")

    fig.suptitle(
        f"{image_path} | raw SAM masks: {result['raw_mask_count']} | device: {result['device']}",
        fontsize=14,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="SAM-based climbing hold detector for verification.")
    parser.add_argument("image", nargs="?", default="IMG_1505.jpeg", help="Path to the wall image.")
    parser.add_argument(
        "--checkpoint",
        default="sam_vit_l_0b3195.pth",
        help="Path to the SAM checkpoint.",
    )
    parser.add_argument("--model-type", default="vit_l", help="SAM model type, for example vit_l.")
    parser.add_argument("--target-count", type=int, default=92, help="Expected number of holds.")
    parser.add_argument("--max-height", type=int, default=1400, help="Resize height for SAM inference.")
    parser.add_argument("--points-per-side", type=int, default=24, help="SAM mask generator density.")
    parser.add_argument("--crop-n-layers", type=int, default=1, help="SAM crop layers.")
    parser.add_argument("--save", help="Optional path to save the debug plot.")
    args = parser.parse_args()

    image_rgb = load_image(args.image)
    result = detect_holds_with_sam(
        image_rgb,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        target_count=args.target_count,
        max_height=args.max_height,
        points_per_side=args.points_per_side,
        crop_n_layers=args.crop_n_layers,
    )

    print(f"Image: {args.image}")
    print(f"Device: {result['device']}")
    print(f"Raw SAM masks: {result['raw_mask_count']}")
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
