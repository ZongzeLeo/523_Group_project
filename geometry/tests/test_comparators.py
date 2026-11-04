"""Analytic numerical checks; synthetic shapes are not clinical validation."""

import numpy as np
import pytest

from geometry.comparators import (GeometryConfig, GeometryError, area_length_volume_ml,
                           disk_volume_ml, ejection_fraction_pct, measure_mask, measure_patient)


def truncated_ellipse(radius_x=20, radius_z=60, base_z=20, spacing=(0.4, 0.5), angle_deg=0):
    # An analytic truncated ellipsoid silhouette, sampled on an anisotropic grid.
    sx, sy = spacing
    h, w = int(180 / sy), int(160 / sx)
    yy, xx = np.indices((h, w))
    x, z = xx * sx - 80, yy * sy - 90
    theta = np.radians(angle_deg)
    xr = np.cos(theta) * x + np.sin(theta) * z
    zr = -np.sin(theta) * x + np.cos(theta) * z
    mask = (xr ** 2 / radius_x ** 2 + zr ** 2 / radius_z ** 2 <= 1) & (zr <= base_z)
    half_base = radius_x * np.sqrt(1 - base_z ** 2 / radius_z ** 2)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = np.array([[-half_base, base_z], [half_base, base_z], [0, -radius_z]]) @ rotation.T + [80, 90]
    landmarks = dict(zip(("left_base", "right_base", "apex"), points / spacing))
    return mask, landmarks


def test_disk_and_area_length_against_analytic_ellipsoid():
    rx, ry, rz = 20, 25, 60
    n = 20
    normalized_z = 2 * (np.arange(n) + 0.5) / n - 1
    d2 = 2 * rx * np.sqrt(1 - normalized_z ** 2)
    d4 = 2 * ry * np.sqrt(1 - normalized_z ** 2)
    expected = 4 / 3 * np.pi * rx * ry * rz / 1000
    assert disk_volume_ml(d2, d4, 2 * rz, 2 * rz) == pytest.approx(expected, rel=0.002)
    assert area_length_volume_ml(np.pi * rx * rz, np.pi * ry * rz, 2 * rz, 2 * rz) == pytest.approx(expected)


def test_area_length_unequal_axis_convention():
    a2, a4, l2, l4 = 3000, 2500, 90, 80
    expected = np.pi / 6 * max(l2, l4) * (4 * a2 / (np.pi * l2)) * (4 * a4 / (np.pi * l4)) / 1000
    assert area_length_volume_ml(a2, a4, l2, l4) == pytest.approx(expected)
    assert area_length_volume_ml(a4, a2, l4, l2) == pytest.approx(expected)


@pytest.mark.parametrize("angle", [-20, 0, 20])
def test_automatic_landmarks_on_rotated_anisotropic_shape(angle):
    spacing = (0.4, 0.5)
    mask, expected = truncated_ellipse(spacing=spacing, angle_deg=angle)
    measured = measure_mask(mask, spacing)
    assert measured["length_mm"] == pytest.approx(80, abs=1.5)
    for key in ("left_base", "right_base", "apex"):
        distance = np.linalg.norm((np.array(measured[key + "_xy_px"]) - expected[key]) * spacing)
        assert distance < 2.0, (key, distance)


def test_native_mask_disk_volume_against_truncated_ellipsoid():
    spacing = (0.4, 0.5)
    a, landmarks_a = truncated_ellipse(20, spacing=spacing)
    b, landmarks_b = truncated_ellipse(25, spacing=spacing)
    ma = measure_mask(a, spacing, landmarks_xy_px=landmarks_a)
    mb = measure_mask(b, spacing, landmarks_xy_px=landmarks_b)
    primitive = lambda z: z - z ** 3 / (3 * 60 ** 2)
    expected = np.pi * 20 * 25 * (primitive(20) - primitive(-60)) / 1000
    assert disk_volume_ml(ma["diameters_mm"], mb["diameters_mm"], ma["length_mm"], mb["length_mm"]) == pytest.approx(expected, rel=0.015)


def test_uniform_physical_scaling_is_cubic():
    d2, d4 = np.arange(1, 21, dtype=float), np.arange(1, 21, dtype=float) * 1.2
    v = disk_volume_ml(d2, d4, 70, 75)
    assert disk_volume_ml(d2 * 2, d4 * 2, 140, 150) == pytest.approx(v * 8)
    a = area_length_volume_ml(1000, 1200, 70, 75)
    assert area_length_volume_ml(4000, 4800, 140, 150) == pytest.approx(a * 8)


def test_translation_preserves_measurements():
    mask, _ = truncated_ellipse()
    shifted = np.pad(mask, ((13, 0), (17, 0)))
    before, after = measure_mask(mask, (0.4, 0.5)), measure_mask(shifted, (0.4, 0.5))
    assert after["length_mm"] == pytest.approx(before["length_mm"])
    assert after["area_mm2"] == pytest.approx(before["area_mm2"])
    np.testing.assert_allclose(after["diameters_mm"], before["diameters_mm"], atol=1e-8)
    np.testing.assert_allclose(np.array(after["apex_xy_px"]) - before["apex_xy_px"], [17, 13])


def test_patient_ef_uses_separate_phase_lengths():
    mask, _ = truncated_ellipse()
    frame = measure_mask(mask, (0.4, 0.5))
    small = dict(frame, length_mm=frame["length_mm"] * 0.8, area_mm2=frame["area_mm2"] * 0.8 ** 2,
                 diameters_mm=(np.array(frame["diameters_mm"]) * 0.8).tolist())
    results = measure_patient({"2CH_ED": frame, "4CH_ED": frame, "2CH_ES": small, "4CH_ES": small})
    for result in results.values():
        assert result["ef_pct"] == pytest.approx(100 * (1 - 0.8 ** 3))
        assert result["status"] == "ok"


def test_invalid_volumes_are_retained_and_flagged():
    mask, _ = truncated_ellipse()
    frame = measure_mask(mask, (0.4, 0.5))
    bigger = dict(frame, length_mm=frame["length_mm"] * 1.2, area_mm2=frame["area_mm2"] * 1.2 ** 2,
                  diameters_mm=(np.array(frame["diameters_mm"]) * 1.2).tolist())
    results = measure_patient({"2CH_ED": frame, "4CH_ED": frame, "2CH_ES": bigger, "4CH_ES": bigger})
    assert all(row["status"] == "invalid_volumes" and row["ef_pct"] < 0 for row in results.values())


@pytest.mark.parametrize("defect", ["empty", "hole", "component", "clipped", "multiclass", "nan"])
def test_invalid_masks_fail_without_cleanup(defect):
    mask, _ = truncated_ellipse()
    mask = mask.astype(float)
    if defect == "empty": mask[:] = 0
    elif defect == "hole": mask[180, 200] = 0
    elif defect == "component": mask[5:8, 5:8] = 1
    elif defect == "clipped": mask[0, :] = 1
    elif defect == "multiclass": mask[10, 10] = 2
    elif defect == "nan": mask[10, 10] = np.nan
    with pytest.raises(GeometryError):
        measure_mask(mask, (0.4, 0.5))


def test_invalid_units_and_inputs():
    mask, _ = truncated_ellipse()
    with pytest.raises(GeometryError): measure_mask(mask, (0, 1))
    with pytest.raises(GeometryError): disk_volume_ml([1, 2], [1], 1, 1)
    with pytest.raises(GeometryError): area_length_volume_ml(np.nan, 1, 1, 1)
    with pytest.raises(GeometryError): ejection_fraction_pct(0, 1)
    with pytest.raises(GeometryError): measure_patient({})
    assert ejection_fraction_pct(150, 60) == pytest.approx(60)


def test_disk_quadrature_converges():
    errors = []
    expected = 4 / 3 * np.pi * 20 * 25 * 60 / 1000
    for n in (10, 20, 40):
        z = 2 * (np.arange(n) + 0.5) / n - 1
        estimate = disk_volume_ml(40 * np.sqrt(1-z*z), 50 * np.sqrt(1-z*z), 120, 120)
        errors.append(abs(estimate - expected))
    assert errors[2] < errors[1] < errors[0]
