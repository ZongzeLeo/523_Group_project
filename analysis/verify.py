"""Verify the published artifact and independently recalculate headline results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12), f"Metric mismatch: {label}")


def verify(root, predictions=None):
    root = Path(root)
    release = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
    for path, expected in release["files_sha256"].items():
        require(digest(root / path) == expected, f"Release file changed: {path}")
    for folder in ("results/geometry", "results/analysis"):
        metadata = json.loads((root / folder / "run_metadata.json").read_text())
        for name, expected in metadata["artifact_sha256"].items():
            require(digest(root / folder / name) == expected, f"Result changed: {folder}/{name}")
    figures = json.loads((root / "results/figures/figure_manifest.json").read_text())
    for name, expected in figures["artifact_sha256"].items():
        require(digest(root / "results/figures" / name) == expected, f"Figure changed: {name}")
    require(digest(root / "analysis/figures.py") == figures["plotting_code_sha256"], "Figure source changed")
    model = root / "segmentation/model_artifact"
    metadata = json.loads((model / "run_metadata.json").read_text())
    compatibility = json.loads((model / "model_compatibility.json").read_text())
    require(digest(model / "weights/best.pt") == metadata["checkpoint_sha256"], "Checkpoint changed")
    require(digest(root / "segmentation/model.py") == compatibility["packaged_model_sha256"], "Model source changed")
    require(metadata["test_frames_used"] == 0, "Test frames used in training")
    expert = read_csv(root / "results/geometry/patients.csv")
    frames = read_csv(root / "results/analysis/frames.csv")
    patients = read_csv(root / "results/analysis/patients.csv")
    require(len(expert) == len(patients) == 50 and len(frames) == 200, "Incomplete test cohort")
    require(len({r['patient_id'] for r in patients}) == 50 and len({r['frame_id'] for r in frames}) == 200,
            "Duplicate test records")
    splits = read_csv(root / "data/reference/splits.csv")
    test_ids = {r['patient_id'] for r in splits if r['split'] == 'test'}
    require({r['patient_id'] for r in patients} == test_ids, "Test split changed")
    require(not test_ids.intersection(metadata['train_patients']+metadata['validation_patients']), "Patient leakage")
    require(all(r['status'] == 'ok' and float(r['ef_pct']) == float(r['reference_ef_pct']) for r in expert),
            "Expert-mask EF reproduction is no longer exact")
    summary = json.loads((root / "results/analysis/summary.json").read_text())
    metric = summary['geometry']['publisher_original']
    dice = statistics.mean(float(r['dice']) for r in frames)
    close(dice, summary['segmentation']['dice_mean'], 'mean Dice')
    for field in ('assd_mm', 'hd95_mm'):
        close(statistics.mean(float(r[field]) for r in frames), summary['segmentation'][field+'_mean'], field)
    errors = [float(r['predicted_ef_pct'])-float(r['camus_reference_ef_pct']) for r in patients]
    mae, bias = statistics.mean(map(abs, errors)), statistics.mean(errors)
    close(mae, metric['ef_vs_camus_pp_mae'], 'EF MAE')
    close(bias, metric['ef_vs_camus_pp_bias'], 'EF bias')
    close(math.sqrt(statistics.mean(v*v for v in errors)), metric['ef_vs_camus_pp_rmse'], 'EF RMSE')
    close(bias-1.96*statistics.stdev(errors), metric['ef_vs_camus_pp_loa_lower'], 'lower LoA')
    close(bias+1.96*statistics.stdev(errors), metric['ef_vs_camus_pp_loa_upper'], 'upper LoA')
    for quantity in ('edv', 'esv'):
        values = [abs(float(r['predicted_'+quantity+'_ml'])-float(r['gt_'+quantity+'_ml'])) for r in patients]
        close(statistics.mean(values), metric[quantity+'_vs_gt_ml_mae'], quantity+' MAE')
    checked_predictions = 0
    if predictions is not None:
        expected = read_csv(root / "results/prediction_manifest.csv")
        actual = read_csv(Path(predictions) / "predictions.csv")
        indexed = {r['frame_id']: r for r in actual}
        require(len(actual) == len(indexed) == 200, 'Incomplete or duplicated prediction records')
        require(set(indexed) == {r['frame_id'] for r in expected}, 'Prediction population changed')
        for row in expected:
            generated = indexed[row['frame_id']]
            require(digest(Path(predictions)/generated['path']) == row['sha256'],
                    f"Prediction bytes differ from the recorded run: {row['frame_id']}")
            checked_predictions += 1
    result = {'status': 'passed', 'verified_release_files': len(release['files_sha256']),
              'expert_exact_matches': 50, 'test_patients': 50, 'native_frames': 200,
              'mean_dice': dice, 'ef_mae_pp': mae, 'prediction_files_checked': checked_predictions}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--predictions', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.predictions), indent=2))
