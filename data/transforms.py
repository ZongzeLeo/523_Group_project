"""Physical-aspect-preserving resampling with explicit pixel-center transforms."""

import numpy as np
from scipy.ndimage import affine_transform


def make_transform(shape_yx, spacing_xy_mm, output_shape_yx=(256, 256)) -> dict:
    h, w = map(int, shape_yx)
    oh, ow = map(int, output_shape_yx)
    sx, sy = map(float, spacing_xy_mm)
    if min(h, w, oh, ow) < 2 or min(sx, sy) <= 0 or not np.isfinite([sx, sy]).all():
        raise ValueError("Invalid dimensions or spacing")
    # Fit the full physical pixel-cell extent into an isotropic output grid.
    spacing = max(w * sx / ow, h * sy / oh)
    scale_x, scale_y = spacing / sx, spacing / sy
    offset_x = (w - 1) / 2 - scale_x * (ow - 1) / 2
    offset_y = (h - 1) / 2 - scale_y * (oh - 1) / 2
    matrix = np.array([[scale_x, 0, offset_x], [0, scale_y, offset_y], [0, 0, 1]])
    return {
        "native_shape_yx": [h, w], "output_shape_yx": [oh, ow],
        "native_spacing_xy_mm": [sx, sy], "output_spacing_xy_mm": [spacing, spacing],
        "model_to_native_xy": matrix.tolist(), "native_to_model_xy": np.linalg.inv(matrix).tolist(),
        "coordinate_convention": "zero-based pixel centers; matrices map [x, y, 1]",
    }


def _resample(array, matrix_xy, output_shape, order):
    matrix = np.asarray(matrix_xy)
    return affine_transform(array, matrix[:2, :2][::-1, ::-1],
                            offset=matrix[:2, 2][::-1], output_shape=tuple(output_shape),
                            order=order, mode="nearest", prefilter=False)


def preprocess(image, labels, spacing_xy_mm, output_shape_yx=(256, 256)):
    transform = make_transform(image.shape, spacing_xy_mm, output_shape_yx)
    matrix = np.array(transform["model_to_native_xy"])
    yy, xx = np.indices(output_shape_yx)
    native_x = matrix[0, 0] * xx + matrix[0, 2]
    native_y = matrix[1, 1] * yy + matrix[1, 2]
    valid = ((native_x >= -0.5) & (native_x < image.shape[1] - 0.5)
             & (native_y >= -0.5) & (native_y < image.shape[0] - 0.5))
    normalized = _resample(image.astype(np.float32) / 255.0, matrix, output_shape_yx, 1)
    lv_mask = _resample((labels == 1).astype(np.uint8), matrix, output_shape_yx, 0)
    normalized[~valid], lv_mask[~valid] = 0, 0
    if not lv_mask.any():
        raise ValueError("Resampling removed the LV mask; choose a larger output shape")
    return normalized.astype(np.float32), lv_mask, valid, transform


def restore_to_native(model_array, transform, *, is_label=False):
    """Map a model probability map (linear) or label map (nearest) to native yx.

    Threshold probabilities AFTER restoration. Resampling is not lossless;
    original expert masks must be loaded directly for geometry baselines.
    """
    model_array = np.asarray(model_array)
    if list(model_array.shape) != transform["output_shape_yx"]:
        raise ValueError("Model array shape does not match its saved transform")
    array = model_array if is_label else model_array.astype(np.float32)
    return _resample(array, transform["native_to_model_xy"], transform["native_shape_yx"],
                     0 if is_label else 1)
