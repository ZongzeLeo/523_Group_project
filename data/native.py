"""Read CAMUS frames without reorienting or modifying their native grid."""

from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def read_config(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Invalid CFG line in {path.name}: {line!r}")
        key, value = (part.strip() for part in line.split(":", 1))
        if key in result:
            raise ValueError(f"Duplicate CFG key {key!r} in {path.name}")
        result[key] = value
    return result


def read_pair(image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if image.GetDimension() not in (2, 3) or (image.GetDimension() == 3 and image.GetSize()[2] != 1):
        raise ValueError("Expected a 2D frame or a single-slice 3D frame, not a cine")
    if image.GetSize() != mask.GetSize():
        raise ValueError("Image/mask sizes differ")
    for key in ("Spacing", "Origin", "Direction"):
        lhs, rhs = getattr(image, "Get" + key)(), getattr(mask, "Get" + key)()
        if len(lhs) != len(rhs) or not np.all(np.isfinite(lhs)) or not np.allclose(lhs, rhs, rtol=0, atol=1e-6):
            raise ValueError(f"Image/mask {key.lower()} mismatch or invalid geometry")
    if any(value <= 0 for value in image.GetSpacing()):
        raise ValueError("Pixel spacing must be positive")
    direction = np.array(image.GetDirection()).reshape(image.GetDimension(), -1)
    if not np.allclose(direction.T @ direction, np.eye(image.GetDimension()), atol=1e-6):
        raise ValueError("Image direction is not orthonormal")
    # Read pixel values through nibabel: SimpleITK's NIfTI reader can silently
    # replace NaN with zero. Preserve the source values until validation.
    # NIfTI storage axes are xy[z]; NumPy image operations below use yx.
    nifti_image, nifti_mask = nib.load(image_path), nib.load(mask_path)
    if any(item.header.get_xyzt_units()[0] != "mm" for item in (nifti_image, nifti_mask)):
        raise ValueError("Expected explicit millimetre NIfTI spatial units")
    pixels = np.asanyarray(nifti_image.dataobj)
    labels = np.asanyarray(nifti_mask.dataobj)
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        pixels, labels = pixels[:, :, 0], labels[:, :, 0]
    pixels, labels = pixels.T, labels.T
    if pixels.ndim != 2 or min(pixels.shape) < 2:
        raise ValueError("Invalid 2D frame shape")
    if pixels.shape != tuple(reversed(image.GetSize()[:2])) or pixels.shape != labels.shape:
        raise ValueError("NIfTI pixel axes do not match the physical image header")
    if not np.all(np.isfinite(pixels)) or pixels.min() < 0 or pixels.max() > 255:
        raise ValueError("Expected finite CAMUS image values in [0, 255]")
    if pixels.max() == pixels.min():
        raise ValueError("Constant image")
    unique = np.unique(labels)
    if not set(unique.tolist()).issubset({0, 1, 2, 3}):
        raise ValueError(f"Invalid CAMUS mask labels: {unique.tolist()}")
    if not np.any(labels == 1):
        raise ValueError("Empty LV cavity (label 1)")
    metadata = {
        "shape_yx": list(pixels.shape), "spacing_xy_mm": list(image.GetSpacing()[:2]),
        "dimension": image.GetDimension(), "origin": list(image.GetOrigin()),
        "spacing": list(image.GetSpacing()), "direction": list(image.GetDirection()),
        "array_axes": "yx", "physical_metadata_axes": "xy[z]", "labels": unique.astype(int).tolist(),
        "spatial_units": "mm",
    }
    return pixels.astype(np.float32), labels.astype(np.uint8), metadata
