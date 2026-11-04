"""Mask-only landmarks, biplane disk summation, and biplane area-length.

All calculations use native physical millimetres. Local planar coordinates are
index-aligned (x right, y down); this is not inter-view spatial registration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, label
from skimage.measure import approximate_polygon, find_contours


from . import GeometryError


@dataclass(frozen=True)
class GeometryConfig:
    n_disks: int = 20
    straight_tolerance_mm: float = 1.0
    max_base_angle_deg: float = 45.0
    basal_search_fraction: float = 0.5
    max_support_outlier_fraction: float = 0.02
    length_mismatch_warning_fraction: float = 0.20

    def __post_init__(self):
        if not isinstance(self.n_disks, int) or self.n_disks < 4:
            raise GeometryError("n_disks must be an integer >= 4")
        values = [self.straight_tolerance_mm, self.max_base_angle_deg, self.basal_search_fraction,
                  self.max_support_outlier_fraction, self.length_mismatch_warning_fraction]
        if not np.isfinite(values).all() or self.straight_tolerance_mm <= 0:
            raise GeometryError("Invalid geometry configuration")
        if not 0 < self.max_base_angle_deg < 90 or not 0 <= self.basal_search_fraction < 1:
            raise GeometryError("Invalid basal-search orientation parameters")
        if not 0 <= self.max_support_outlier_fraction < 0.5 or self.length_mismatch_warning_fraction <= 0:
            raise GeometryError("Invalid QC thresholds")


def _validate_mask(mask, spacing_xy_mm):
    mask = np.asarray(mask)
    spacing = np.asarray(spacing_xy_mm, dtype=float)
    if mask.ndim != 2 or min(mask.shape) < 3 or not np.isin(mask, [0, 1]).all():
        raise GeometryError("Expected a finite binary 2D LV cavity mask")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise GeometryError("spacing_xy_mm must contain two positive finite values")
    mask = mask.astype(bool)
    if mask.sum() < 10:
        raise GeometryError("Empty or insufficient LV mask")
    if mask[[0, -1], :].any() or mask[:, [0, -1]].any():
        raise GeometryError("LV mask touches the image border; cavity may be clipped")
    _, components = label(mask, structure=np.ones((3, 3)))
    if components != 1:
        raise GeometryError(f"Expected one LV component; found {components}. No component cleanup was applied.")
    holes = int((binary_fill_holes(mask) & ~mask).sum())
    if holes:
        raise GeometryError(f"LV mask contains {holes} hole pixels. No hole filling was applied.")
    return mask, spacing


def _contour(mask, spacing):
    contours = find_contours(mask.astype(np.uint8), 0.5, fully_connected="high")
    if len(contours) != 1:
        raise GeometryError(f"Expected one closed cavity contour; found {len(contours)}")
    contour = contours[0][:, ::-1] * spacing
    if not np.allclose(contour[0], contour[-1]):
        raise GeometryError("Open cavity contour")
    contour = contour[:-1]
    # Start simplification at the superior boundary, away from the basal chord.
    contour = np.roll(contour, -np.argmin(contour[:, 1]), axis=0)
    return np.vstack([contour, contour[0]])


def detect_landmarks(contour_xy_mm, config=GeometryConfig()):
    """Estimate the straight basal closure and farthest contour point (apex).

    Assumes the apical end is superior and the basal closure is in the inferior
    half of the upright CAMUS view. These are mask-derived annular surrogates,
    not independently annotated anatomical landmarks.
    """
    contour = np.asarray(contour_xy_mm)
    polygon = approximate_polygon(contour, tolerance=config.straight_tolerance_mm)
    low_y, high_y = contour[:, 1].min(), contour[:, 1].max()
    candidates = []
    for first, second in zip(polygon[:-1], polygon[1:]):
        left, right = (first, second) if first[0] < second[0] else (second, first)
        delta = right - left
        width = float(np.linalg.norm(delta))
        if width <= 2 * config.straight_tolerance_mm:
            continue
        angle = float(np.degrees(np.arctan2(abs(delta[1]), abs(delta[0]))))
        midpoint = (left + right) / 2
        if angle > config.max_base_angle_deg or midpoint[1] < low_y + config.basal_search_fraction * (high_y - low_y):
            continue
        normal = np.array([-delta[1], delta[0]]) / width  # points inferiorly
        below_fraction = float(np.mean((contour - left) @ normal > 2 * config.straight_tolerance_mm))
        if below_fraction > config.max_support_outlier_fraction:
            continue
        candidates.append((width, left, right, angle, below_fraction))
    if not candidates:
        raise GeometryError("No supported inferior basal chord found under the upright-view assumption")
    width, left, right, angle, below_fraction = max(candidates, key=lambda candidate: candidate[0])
    midpoint = (left + right) / 2
    apex = contour[np.argmax(np.linalg.norm(contour - midpoint, axis=1))]
    axis = apex - midpoint
    length = float(np.linalg.norm(axis))
    if length <= 0 or axis[1] >= 0 or -axis[1] / length < 0.5:
        raise GeometryError("Detected long axis is inconsistent with the upright apical-view assumption")
    return left, right, apex, {"basal_width_mm": width, "basal_angle_deg": angle,
                               "basal_support_outlier_fraction": below_fraction,
                               "basal_candidate_count": len(candidates)}


def sample_diameters(contour, base_midpoint, apex, n_disks):
    axis = apex - base_midpoint
    length = float(np.linalg.norm(axis))
    along = axis / length
    across = np.array([-along[1], along[0]])
    z = (contour - base_midpoint) @ along
    q = (contour - base_midpoint) @ across
    diameters, segments, warnings = [], [], []
    for i in range(n_disks):
        height = length * (i + 0.5) / n_disks
        crosses = ((z[:-1] <= height) & (z[1:] > height)) | ((z[1:] <= height) & (z[:-1] > height))
        crossings = q[:-1][crosses] + (height - z[:-1][crosses]) * (q[1:][crosses] - q[:-1][crosses]) / (z[1:][crosses] - z[:-1][crosses])
        crossings = np.unique(np.round(crossings, decimals=10))
        if len(crossings) < 2:
            raise GeometryError(f"No complete cavity chord at disk {i + 1}/{n_disks}")
        if len(crossings) > 2:
            warnings.append("multiple_intervals_on_disk_line")
        lower, upper = crossings[0], crossings[-1]
        if lower > 0 or upper < 0:
            warnings.append("long_axis_outside_cavity_on_disk_line")
        center = base_midpoint + height * along
        segments.append([center + lower * across, center + upper * across])
        diameters.append(float(upper - lower))
    return np.array(diameters), np.array(segments), sorted(set(warnings))


def measure_mask(mask, spacing_xy_mm, config=GeometryConfig(), *, landmarks_xy_px=None):
    """Return serializable native-grid measurements; optional landmarks bypass detection.

    Explicit landmarks require left_base, right_base, apex in (x,y) pixel-center
    coordinates, for analytic tests or independently provided landmarks. Image
    and reference EF values never enter this computation.
    """
    mask, spacing = _validate_mask(mask, spacing_xy_mm)
    contour = _contour(mask, spacing)
    if landmarks_xy_px is None:
        left, right, apex, diagnostics = detect_landmarks(contour, config)
        landmark_source = "automatic_mask_basal_closure"
    else:
        points = np.asarray([landmarks_xy_px[key] for key in ("left_base", "right_base", "apex")], dtype=float)
        if points.shape != (3, 2) or not np.isfinite(points).all():
            raise GeometryError("Invalid explicit landmark coordinates")
        left, right, apex = points * spacing
        diagnostics = {"basal_width_mm": float(np.linalg.norm(right - left)),
                       "basal_angle_deg": float(np.degrees(np.arctan2(abs(right[1] - left[1]), abs(right[0] - left[0])))),
                       "basal_support_outlier_fraction": None, "basal_candidate_count": None}
        landmark_source = "provided_coordinates"
    midpoint = (left + right) / 2
    length = float(np.linalg.norm(apex - midpoint))
    if length <= 0 or diagnostics["basal_width_mm"] <= 0:
        raise GeometryError("Degenerate landmark geometry")
    diameters, segments, warnings = sample_diameters(contour, midpoint, apex, config.n_disks)
    if diagnostics["basal_width_mm"] > length:
        warnings.append("basal_width_exceeds_long_axis")
    return {
        "area_mm2": float(mask.sum() * spacing.prod()), "length_mm": length,
        "spacing_xy_mm": spacing.tolist(), "shape_yx": list(mask.shape),
        "left_base_xy_px": (left / spacing).tolist(), "right_base_xy_px": (right / spacing).tolist(),
        "base_midpoint_xy_px": (midpoint / spacing).tolist(), "apex_xy_px": (apex / spacing).tolist(),
        "left_base_xy_mm": left.tolist(), "right_base_xy_mm": right.tolist(),
        "base_midpoint_xy_mm": midpoint.tolist(), "apex_xy_mm": apex.tolist(),
        "diameters_mm": diameters.tolist(), "disk_segments_xy_mm": segments.tolist(),
        "landmark_source": landmark_source, "warnings": sorted(set(warnings)), **diagnostics,
    }


def disk_volume_ml(diameters_2ch_mm, diameters_4ch_mm, length_2ch_mm, length_4ch_mm):
    d2, d4 = np.asarray(diameters_2ch_mm, dtype=float), np.asarray(diameters_4ch_mm, dtype=float)
    if d2.ndim != 1 or d2.shape != d4.shape or not len(d2) or not np.isfinite([*d2, *d4, length_2ch_mm, length_4ch_mm]).all():
        raise GeometryError("Invalid paired disk profiles")
    if (d2 <= 0).any() or (d4 <= 0).any() or min(length_2ch_mm, length_4ch_mm) <= 0:
        raise GeometryError("Disk dimensions must be positive")
    height = max(length_2ch_mm, length_4ch_mm) / len(d2)
    return float(np.pi / 4 * height * np.dot(d2, d4) / 1000)


def area_length_volume_ml(area_2ch_mm2, area_4ch_mm2, length_2ch_mm, length_4ch_mm):
    values = np.asarray([area_2ch_mm2, area_4ch_mm2, length_2ch_mm, length_4ch_mm], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise GeometryError("Area-length inputs must be positive and finite")
    # Ellipsoid widths 4*A/(pi*L), with the longer observed axis as 3D length.
    # This simplifies to 8*A2*A4/(3*pi*min(L2,L4)); it is explicitly biplane.
    return float(8 / (3 * np.pi) * area_2ch_mm2 * area_4ch_mm2 / min(length_2ch_mm, length_4ch_mm) / 1000)


def ejection_fraction_pct(edv_ml, esv_ml):
    if not np.isfinite([edv_ml, esv_ml]).all() or edv_ml <= 0 or esv_ml < 0:
        raise GeometryError("Invalid EDV/ESV")
    return float(100 * (edv_ml - esv_ml) / edv_ml)  # retain negative values for explicit QC


def measure_patient(frames, config=GeometryConfig()):
    """Combine four measured frames; no reference values are accepted."""
    expected = {f"{view}_{phase}" for view in ("2CH", "4CH") for phase in ("ED", "ES")}
    if set(frames) != expected:
        raise GeometryError(f"Expected exactly four paired frames: {sorted(expected)}")
    values = {"simpson_biplane": {}, "area_length_biplane_ellipsoid": {}}
    warnings = sorted({warning for frame in frames.values() for warning in frame["warnings"]})
    for phase in ("ED", "ES"):
        a, b = frames[f"2CH_{phase}"], frames[f"4CH_{phase}"]
        l2, l4 = a["length_mm"], b["length_mm"]
        if len(a["diameters_mm"]) != config.n_disks or len(b["diameters_mm"]) != config.n_disks:
            raise GeometryError("Disk profiles do not match the configured count")
        if abs(l2 - l4) / max(l2, l4) > config.length_mismatch_warning_fraction:
            warnings.append(f"{phase.lower()}_view_length_mismatch")
        values["simpson_biplane"][phase] = disk_volume_ml(a["diameters_mm"], b["diameters_mm"], l2, l4)
        values["area_length_biplane_ellipsoid"][phase] = area_length_volume_ml(a["area_mm2"], b["area_mm2"], l2, l4)
    result = {}
    for method, phases in values.items():
        edv, esv = phases["ED"], phases["ES"]
        result[method] = {"edv_ml": edv, "esv_ml": esv, "ef_pct": ejection_fraction_pct(edv, esv),
                          "status": "invalid_volumes" if esv > edv else "ok",
                          "warnings": sorted(set(warnings + (["esv_exceeds_edv"] if esv > edv else [])))}
    return result
