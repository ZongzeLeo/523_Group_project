"""Explicit mask cleanup and native physical boundary metrics."""

import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, label


def clean_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    components, count = label(mask, structure=np.ones((3, 3)))
    if count:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest = components == int(sizes.argmax())
        cleaned = binary_fill_holes(largest)
    else:
        largest = cleaned = mask
    return cleaned.astype(np.uint8), {
        "raw_components": count, "removed_component_pixels": int(mask.sum()-largest.sum()),
        "filled_hole_pixels": int(cleaned.sum()-largest.sum()),
        "empty_prediction": not bool(cleaned.any()),
        "touches_border": bool(cleaned[[0, -1], :].any() or cleaned[:, [0, -1]].any())}


def mask_metrics(predicted, reference, spacing_xy_mm):
    predicted, reference = np.asarray(predicted, dtype=bool), np.asarray(reference, dtype=bool)
    spacing = np.asarray(spacing_xy_mm, dtype=float)
    if predicted.ndim != 2 or predicted.shape != reference.shape:
        raise ValueError("Masks must share the native 2D grid")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Positive physical xy spacing is required")
    p, g = int(predicted.sum()), int(reference.sum())
    intersection = int((predicted & reference).sum())
    result = {"dice": 2*intersection/(p+g) if p+g else 1.0,
              "iou": intersection/(p+g-intersection) if p+g else 1.0,
              "predicted_area_mm2": float(p*spacing.prod()), "gt_area_mm2": float(g*spacing.prod()),
              "area_error_mm2": float((p-g)*spacing.prod()), "assd_mm": None, "hd95_mm": None}
    if not p or not g:
        return result
    ps = predicted & ~binary_erosion(predicted)
    gs = reference & ~binary_erosion(reference)
    distances = np.concatenate([distance_transform_edt(~gs, sampling=spacing[::-1])[ps],
                                distance_transform_edt(~ps, sampling=spacing[::-1])[gs]])
    result.update(assd_mm=float(distances.mean()), hd95_mm=float(np.percentile(distances, 95)))
    return result
