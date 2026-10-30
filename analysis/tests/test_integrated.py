import numpy as np
import pytest

from analysis.run import index_predictions
from analysis.statistics import agreement, bootstrap_mean_ci, geometry_metrics


def test_pairing_rejects_duplicates_missing_frames_and_wrong_phase():
    frames = [{"frame_id": "p_2CH_ED", "patient_id": "p", "view": "2CH", "phase": "ED", "split": "test"}]
    assert index_predictions(frames, frames)["p_2CH_ED"] == frames[0]
    with pytest.raises(ValueError): index_predictions(frames, frames*2)
    with pytest.raises(ValueError): index_predictions(frames, [])
    with pytest.raises(ValueError): index_predictions(frames, [dict(frames[0], phase="ES")])


def test_agreement_and_patient_bootstrap_known_values():
    result = agreement([2, -2, 0])
    assert result['mae'] == pytest.approx(4/3)
    assert result['bias'] == 0 and result['loa_upper'] == pytest.approx(3.92)
    assert bootstrap_mean_ci([7, 7, 7]) == [7, 7]
    assert bootstrap_mean_ci([1, 2, 4]) == bootstrap_mean_ci([1, 2, 4])
    with pytest.raises(ValueError): bootstrap_mean_ci([np.nan])


def test_invalid_negative_ef_is_not_removed_from_error_metrics():
    row = {'status': 'invalid_volumes', 'predicted_ef_pct': -10, 'camus_reference_ef_pct': 50,
           'warnings': ['esv_exceeds_edv'], 'ef_error_vs_camus_pp': -60, 'ef_error_vs_gt_pp': -60,
           'edv_error_vs_gt_ml': 20, 'esv_error_vs_gt_ml': 30}
    missing = dict(row, status='invalid_geometry', predicted_ef_pct=None)
    result = geometry_metrics([row, missing])
    assert result['attempted_patients'] == 2 and result['numeric_pairs'] == 1
    assert result['invalid_volume_patients'] == result['failed_geometry_patients'] == 1
    assert result['ef_vs_camus_pp_mae'] == 60
