import numpy as np
import pytest

from segmentation.metrics import clean_mask, mask_metrics


def test_identical_masks_and_anisotropic_surface_distance():
    mask = np.zeros((10, 12), dtype=bool)
    mask[4, 4] = True
    same = mask_metrics(mask, mask, [2, 3])
    assert same['dice'] == 1 and same['assd_mm'] == same['hd95_mm'] == 0
    shifted = np.zeros_like(mask)
    shifted[4, 6] = True
    metrics = mask_metrics(shifted, mask, [2, 3])
    assert metrics['dice'] == 0
    assert metrics['assd_mm'] == metrics['hd95_mm'] == 4
    assert metrics['gt_area_mm2'] == 6


def test_explicit_cleanup_counts_components_and_holes():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    mask[9, 9] = False
    mask[1, 1] = True
    cleaned, info = clean_mask(mask)
    assert cleaned.sum() == 100
    assert info['raw_components'] == 2
    assert info['removed_component_pixels'] == info['filled_hole_pixels'] == 1
    assert not info['empty_prediction'] and not info['touches_border']


def test_empty_prediction_has_zero_dice_and_unavailable_distances():
    gt = np.ones((12, 12), dtype=bool)
    empty = np.zeros_like(gt)
    metric = mask_metrics(empty, gt, [.5, .8])
    assert metric['dice'] == 0 and metric['assd_mm'] is metric['hd95_mm'] is None
    assert clean_mask(empty)[1]['empty_prediction']
    with pytest.raises(ValueError): mask_metrics(empty, gt, [0, 1])
