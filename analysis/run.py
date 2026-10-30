"""Evaluate native LV masks and propagate their errors through publisher geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from data.dataset import CamusDataset
from data.download_camus import atomic_json, sha256
from data.prepare import write_csv
from data.protocol import identity, verify_inputs, verify_results
from geometry import GeometryError
from geometry.publisher import measure_publisher, NOTEBOOK_SHA256
from segmentation.metrics import clean_mask, mask_metrics
from .statistics import bootstrap_mean_ci, stratify_frames, stratify_geometry


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def index_predictions(expected_frames, rows):
    expected = {row["frame_id"]: row for row in expected_frames}
    actual = {row["frame_id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("Prediction IDs must match every selected native frame exactly once")
    for frame_id, row in actual.items():
        if any(row[field] != expected[frame_id][field] for field in ("patient_id", "split", "view", "phase")):
            raise ValueError(f"Prediction pairing metadata mismatch: {frame_id}")
    return actual


@lru_cache(maxsize=1)
def _inputs(prepared, predictions, expert, split):
    dataset = CamusDataset(prepared, split)
    index = index_predictions(dataset.frames, read_csv(Path(predictions) / "predictions.csv"))
    gt = {row["patient_id"]: row for row in read_csv(Path(expert) / "patients.csv")}
    if set(gt) != set(dataset.patient_metadata):
        raise ValueError("Expert geometry must contain every selected patient exactly once")
    return dataset, index, gt


def evaluate_patient(task):
    prepared, predictions, expert, split, patient_id, threshold = task
    dataset, index, references = _inputs(prepared, predictions, expert, split)
    native = dataset.patient_frames(patient_id)
    patient = dataset.patient_metadata[patient_id]
    quality_rank = {"Good": 0, "Medium": 1, "Poor": 2}
    quality = max([patient["quality_2ch"], patient["quality_4ch"]], key=quality_rank.__getitem__)
    predicted_native, frames = {}, []
    for key, frame in native.items():
        record = index[frame["frame_id"]]
        path = Path(predictions) / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Native prediction changed: {frame['frame_id']}")
        with np.load(path, allow_pickle=False) as arrays:
            probability, raw, mask = arrays["probability"], arrays["raw_mask"], arrays["mask"]
        if (probability.shape != frame["mask"].shape or not np.isfinite(probability).all()
                or probability.min() < 0 or probability.max() > 1):
            raise ValueError("Prediction probability does not match the native image grid")
        cleaned, flags = clean_mask(raw)
        if not np.array_equal(raw, probability > threshold) or not np.array_equal(mask, cleaned):
            raise ValueError("Saved masks do not follow the frozen threshold/cleanup rule")
        predicted_native[key] = dict(frame, mask=mask)
        spacing = frame["geometry"]["spacing_xy_mm"]
        frames.append({"frame_id": frame["frame_id"], "patient_id": patient_id, "split": split,
                       "view": frame["view"], "phase": frame["phase"], "view_phase": key,
                       "image_quality": patient[f"quality_{frame['view'].lower()}"],
                       "reference_ef_group": patient["reference_ef_group"],
                       **mask_metrics(mask, frame["mask"], spacing),
                       "raw_dice": mask_metrics(raw, frame["mask"], spacing)["dice"], **flags})
    error = ""
    try:
        prediction = measure_publisher(predicted_native)
    except GeometryError as exc:
        error = str(exc)
        prediction = {"edv_ml": None, "esv_ml": None, "ef_pct": None,
                      "status": "invalid_geometry", "warnings": []}
    gt = references[patient_id]
    if gt["status"] != "ok":
        raise ValueError("A valid expert-mask reference is required for every patient")
    row = {"patient_id": patient_id, "split": split, "method": "publisher_original",
           "reference_ef_group": patient["reference_ef_group"], "image_quality": quality,
           "status": prediction["status"], "warnings": prediction["warnings"], "error": error,
           "camus_reference_ef_pct": float(patient["reference_ef_pct"]),
           "mean_frame_dice": float(np.mean([r["dice"] for r in frames])),
           "mean_raw_frame_dice": float(np.mean([r["raw_dice"] for r in frames]))}
    for quantity in ("edv_ml", "esv_ml", "ef_pct"):
        row[f"predicted_{quantity}"] = prediction[quantity]
        row[f"gt_{quantity}"] = float(gt[quantity])
    for quantity, unit in (("edv", "ml"), ("esv", "ml"), ("ef", "pp")):
        key = f"{quantity}_{'pct' if quantity == 'ef' else 'ml'}"
        row[f"{quantity}_error_vs_gt_{unit}"] = prediction[key]-float(gt[key]) if prediction[key] is not None else None
    row["ef_error_vs_camus_pp"] = prediction["ef_pct"]-row["camus_reference_ef_pct"] if prediction["ef_pct"] is not None else None
    return frames, row


def evaluate(prepared, predictions, expert, output, split="test", workers=4):
    prepared, predictions, expert, output = map(Path, (prepared, predictions, expert, output))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output is nonempty; use a new directory")
    verify_inputs(prepared)
    pm = json.loads((predictions / "run_metadata.json").read_text())
    if pm["status"] != "complete" or pm["split"] != split:
        raise ValueError("Incomplete or mismatched prediction split")
    if sha256(predictions / "predictions.csv") != pm["prediction_manifest_sha256"]:
        raise ValueError("Prediction manifest changed")
    gm = verify_results(expert)
    if (gm["data_identity"] != identity(prepared)
            or pm["prepared_metadata_sha256"] != sha256(prepared / "run_metadata.json")):
        raise ValueError("Expert reference and predictions have different data provenance")
    dataset = CamusDataset(prepared, split)
    index_predictions(dataset.frames, read_csv(predictions / "predictions.csv"))
    if set(pm["patients"]) != set(dataset.patient_metadata):
        raise ValueError("Prediction patients differ from the official split")
    args = tuple(str(path.resolve()) for path in (prepared, predictions, expert))
    tasks = [(*args, split, patient, pm["threshold"]) for patient in dataset.patient_metadata]
    frames, patients = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for frame_rows, patient in pool.map(evaluate_patient, tasks):
            frames.extend(frame_rows)
            patients.append(patient)
    geometry, segmentation = stratify_geometry(patients), stratify_frames(frames)
    summary = {"split": split, "patients": len(patients), "frames": len(frames),
               "segmentation": segmentation[0],
               "mean_dice_patient_bootstrap_ci95": bootstrap_mean_ci([r["mean_frame_dice"] for r in patients]),
               "geometry": {"publisher_original": geometry[0]},
               "expert_publisher_exact_matches": sum(r["gt_ef_pct"] == r["camus_reference_ef_pct"] for r in patients),
               "interpretation": ["EF follows the original publisher calculation, including integer rounding.",
                                  "Volume errors use expert-mask-derived volumes under the identical calculation.",
                                  "Finite negative EF remains in error metrics; invalid volumes and failures are counted.",
                                  "Confidence intervals resample patients: 2000 replicates, seed 523."]}
    output.mkdir(parents=True)
    write_csv(output / "frames.csv", frames)
    write_csv(output / "patients.csv", [dict(row, warnings=";".join(row["warnings"])) for row in patients])
    write_csv(output / "segmentation_metrics.csv", segmentation)
    write_csv(output / "geometry_metrics.csv", geometry)
    atomic_json(output / "summary.json", summary)
    source = ["analysis/run.py", "analysis/statistics.py", "segmentation/metrics.py", "geometry/publisher.py"]
    artifacts = ["frames.csv", "patients.csv", "segmentation_metrics.csv", "geometry_metrics.csv", "summary.json"]
    atomic_json(output / "run_metadata.json", {
        "status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_metadata_sha256": sha256(prepared / "run_metadata.json"), "data_identity": identity(prepared),
        "prediction_metadata_sha256": sha256(predictions / "run_metadata.json"),
        "checkpoint_sha256": pm["checkpoint_sha256"], "publisher_notebook_sha256": NOTEBOOK_SHA256,
        "expert_metadata_sha256": sha256(expert / "run_metadata.json"),
        "code_sha256": {name: sha256(Path(name)) for name in source},
        "artifact_sha256": {name: sha256(output / name) for name in artifacts}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--expert", type=Path, default=Path("outputs/geometry"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = evaluate(args.prepared, args.predictions, args.expert, args.output, args.split, args.workers)
    print(json.dumps({"patients": result["patients"], "dice": result["segmentation"]["dice_mean"],
                      "ef_mae_pp": result["geometry"]["publisher_original"]["ef_vs_camus_pp_mae"]}, indent=2))
