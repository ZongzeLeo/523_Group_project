"""Physical intervention checks independent of measured EF outcomes."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.perturbations import axial_coordinates, perturb_mask, scenarios
from geometry import GeometryError


def rectangle(spacing=(0.5, 0.25)):
    yy, xx = np.indices((500, 240))
    mask = ((xx*spacing[0] >= 40) & (xx*spacing[0] <= 80)
            & (yy*spacing[1] >= 20) & (yy*spacing[1] <= 100)).astype(np.uint8)
    landmarks = {"base_midpoint_xy_mm": [60., 100.], "apex_xy_mm": [60., 20.]}
    return mask, landmarks


@pytest.mark.parametrize("family", ["control", "apical_truncation", "basal_displacement"])
def test_zero_is_bitwise_identity_and_does_not_mutate(family):
    mask, landmarks = rectangle()
    original = mask.copy()
    changed, audit = perturb_mask(mask, (0.5, 0.25), landmarks, family, 0)
    np.testing.assert_array_equal(changed, original)
    np.testing.assert_array_equal(mask, original)
    assert audit["changed_pixels"] == 0
    assert not np.shares_memory(mask, changed)


def test_apical_cut_has_requested_physical_depth():
    mask, landmarks = rectangle()
    changed, audit = perturb_mask(mask, (0.5, 0.25), landmarks, "apical_truncation", 2)
    assert audit["apical_extent_change_mm"] == -2
    assert audit["basal_extent_change_mm"] == 0
    assert np.all(changed <= mask)
    assert audit["changed_pixels"] == 8*81


@pytest.mark.parametrize("displacement", [-2., -1., 1., 2.])
def test_basal_direction_and_local_support(displacement):
    mask, landmarks = rectangle()
    changed, audit = perturb_mask(mask, (0.5, 0.25), landmarks, "basal_displacement", displacement)
    assert audit["basal_extent_change_mm"] == displacement
    assert audit["apical_extent_change_mm"] == 0
    z, _, _ = axial_coordinates(mask.shape, (0.5, 0.25), landmarks)
    np.testing.assert_array_equal(changed[z >= 10], mask[z >= 10])
    assert not audit["touches_border"]


def test_physical_scaling_preserves_intervention_pixels():
    mask, landmarks = rectangle()
    scaled = {k: (np.array(v)*2).tolist() for k, v in landmarks.items()}
    for family in ("apical_truncation", "basal_displacement"):
        a, _ = perturb_mask(mask, (0.5, 0.25), landmarks, family, 1, 10)
        b, _ = perturb_mask(mask, (1., 0.5), scaled, family, 2, 20)
        np.testing.assert_array_equal(a, b)


def test_rotated_axis_uses_physical_coordinates():
    yy, xx = np.indices((240, 240))
    spacing = (0.6, 0.7)
    base, apex = np.array([85., 130.]), np.array([55., 50.])
    landmarks = {"base_midpoint_xy_mm": base.tolist(), "apex_xy_mm": apex.tolist()}
    axis = (apex-base)/np.linalg.norm(apex-base)
    z = (xx*spacing[0]-base[0])*axis[0]+(yy*spacing[1]-base[1])*axis[1]
    q = -(xx*spacing[0]-base[0])*axis[1]+(yy*spacing[1]-base[1])*axis[0]
    mask = (abs(q) < 20) & (z >= 0) & (z <= np.linalg.norm(apex-base))
    changed, _ = perturb_mask(mask, spacing, landmarks, "apical_truncation", 2)
    assert z[changed.astype(bool)].max() <= np.linalg.norm(apex-base)-2
    np.testing.assert_array_equal(changed[z < 40], mask[z < 40])


def test_complete_pairing_matrix():
    config = json.loads((Path(__file__).parents[1]/"configs/perturbations_v1.json").read_text())
    matrix = scenarios(config)
    assert len(matrix) == 55 and len({s.scenario_id for s in matrix}) == 55
    assert not any(matrix[0].affects(k) for k in ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"))
    case = next(s for s in matrix if s.scenario_id == "apical_truncation_p1mm_2CH_ES")
    assert [k for k in ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES") if case.affects(k)] == ["2CH_ES"]


def test_invalid_warp_rejected_without_cleanup():
    mask, landmarks = rectangle()
    with pytest.raises(GeometryError):
        perturb_mask(mask, (0.5, 0.25), landmarks, "basal_displacement", 10, 10)
    with pytest.raises(GeometryError):
        perturb_mask(mask, (0.5, 0.25), landmarks, "apical_truncation", -1)
