"""Run the frozen perturbation matrix and explain saved segmentation errors."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import skimage

from data.dataset import CamusDataset
from data.download_camus import atomic_json, sha256
from data.protocol import identity, verify_inputs
from geometry import GeometryError
from geometry.comparators import measure_mask, measure_patient
from geometry.publisher import measure_publisher, NOTEBOOK_SHA256
from geometry.publisher_profiles import publisher_profile, measure_profiles
from segmentation.metrics import clean_mask, mask_metrics
from .perturbations import METHODS, FRAME_KEYS, perturb_mask, scenarios
from .run import index_predictions, read_csv
from .sensitivity_statistics import (method_contrasts, observed_summaries, paired_summary,
                                     volume_error_contributions)

ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def failed(error):
    return {"edv_ml": None, "esv_ml": None, "ef_pct": None, "status": "invalid_geometry",
            "warnings": [], "error": str(error)}


def all_methods(native, measured, profiles=None):
    results = {}
    try:
        results["publisher_original"] = measure_profiles(profiles) if profiles is not None else measure_publisher(native)
    except GeometryError as exc:
        results["publisher_original"] = failed(exc)
    try:
        if any(v is None for v in measured.values()):
            raise GeometryError("At least one frame failed secondary geometry")
        results.update(measure_patient(measured))
    except GeometryError as exc:
        for method in METHODS[1:]:
            results[method] = failed(exc)
    return results


def geometry_differences(measured, reference):
    row = {"length_error_mm": None, "area_error_pct": None, "abs_length_error_mm": None,
           "abs_area_error_pct": None, "apex_displacement_mm": None, "base_midpoint_displacement_mm": None}
    if measured is None:
        return row
    row["length_error_mm"] = measured["length_mm"]-reference["length_mm"]
    row["area_error_pct"] = 100*(measured["area_mm2"]-reference["area_mm2"])/reference["area_mm2"]
    row["abs_length_error_mm"] = abs(row["length_error_mm"])
    row["abs_area_error_pct"] = abs(row["area_error_pct"])
    axis = np.asarray(reference["apex_xy_mm"])-reference["base_midpoint_xy_mm"]
    axis /= np.linalg.norm(axis)
    for point in ("apex", "base_midpoint", "left_base", "right_base"):
        delta = np.asarray(measured[point+"_xy_mm"])-reference[point+"_xy_mm"]
        row[point+"_displacement_mm"] = float(np.linalg.norm(delta))
        row[point+"_axial_displacement_mm"] = float(delta@axis)
        for kind, measurement in (("gt", reference), ("measured", measured)):
            for i, dim in enumerate(("x", "y")):
                row[f"{kind}_{point}_{dim}_mm"] = measurement[point+"_xy_mm"][i]
    return row


@lru_cache(maxsize=1)
def inputs(prepared, predictions):
    dataset = CamusDataset(prepared, "test")
    return dataset, index_predictions(dataset.frames, read_csv(Path(predictions)/"predictions.csv"))


def patient_experiment(task):
    prepared, predictions, patient_id, config = task
    dataset, prediction_index = inputs(prepared, predictions)
    native = dataset.patient_frames(patient_id)
    patient = dataset.patient_metadata[patient_id]
    quality_rank = {"Good": 0, "Medium": 1, "Poor": 2}
    attributes = {"patient_id": patient_id, "split": "test", "reference_ef_group": patient["reference_ef_group"],
                  "image_quality": max([patient["quality_2ch"], patient["quality_4ch"]], key=quality_rank.__getitem__)}
    reference = {k: measure_mask(f["mask"], f["geometry"]["spacing_xy_mm"]) for k, f in native.items()}
    baseline = all_methods(native, reference)
    if any(r["status"] != "ok" for r in baseline.values()):
        raise ValueError(f"Invalid unchanged expert baseline: {patient_id}")
    if baseline["publisher_original"]["ef_pct"] != float(patient["reference_ef_pct"]):
        raise ValueError(f"Expert EF reproduction changed: {patient_id}")
    cache, frame_audits = {}, []
    interventions = [("control", 0.)]+[("apical_truncation", d) for d in config["apical_truncation_mm"]]
    interventions += [("basal_displacement", d) for d in config["basal_displacement_mm"]]
    for key, frame in native.items():
        spacing = frame["geometry"]["spacing_xy_mm"]
        for family, value in interventions:
            mask, audit = perturb_mask(frame["mask"], spacing, reference[key], family, value, config["basal_taper_mm"])
            geometry, error = None, ""
            try:
                geometry = reference[key] if family == "control" else measure_mask(mask, spacing)
            except GeometryError as exc:
                error = str(exc)
            record = {**attributes, "frame_id": frame["frame_id"], "view_phase": key,
                      "family": family, "magnitude_mm": value, **audit,
                      "geometry_status": "ok" if geometry is not None else "invalid_geometry", "geometry_error": error,
                      **geometry_differences(geometry, reference[key]),
                      **{k: v for k, v in mask_metrics(mask, frame["mask"], spacing).items()
                         if k in ("dice", "assd_mm", "hd95_mm")}}
            frame_audits.append(record)
            changed_frame = dict(frame, mask=mask)
            try:
                profile = publisher_profile(changed_frame)
                record["publisher_profile_error"] = ""
            except GeometryError as exc:
                profile = None
                record["publisher_profile_error"] = str(exc)
            cache[key, family, value] = (changed_frame, geometry, audit, profile)
    simulated = []
    for scenario in scenarios(config):
        changed, measured, profiles, border = {}, {}, {}, False
        for key in FRAME_KEYS:
            family, value = (scenario.family, scenario.magnitude_mm) if scenario.affects(key) else ("control", 0.)
            changed[key], measured[key], audit, profiles[key] = cache[key, family, value]
            border = border or audit["touches_border"]
        results = all_methods(changed, measured, profiles)
        if scenario.family == "control":
            for field in ("edv_ml", "esv_ml", "ef_pct"):
                if results["publisher_original"][field] != baseline["publisher_original"][field]:
                    raise ValueError("Cached and original publisher baseline differ")
        for method in METHODS:
            estimate, base = results[method], baseline[method]
            row = {**attributes, **scenario.record(), "method": method,
                   "status": estimate["status"], "border_contact": border,
                   "warnings": ";".join(estimate["warnings"]), "error": estimate.get("error", "")}
            for field, unit in (("edv_ml", "edv_ml"), ("esv_ml", "esv_ml"), ("ef_pct", "ef_pp")):
                row["baseline_"+field], row["perturbed_"+field] = base[field], estimate[field]
                row["delta_"+unit] = estimate[field]-base[field] if estimate[field] is not None else None
            simulated.append(row)

    predicted, measured, observed_frames = {}, {}, []
    for key, frame in native.items():
        record = prediction_index[frame["frame_id"]]
        path = Path(predictions)/record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Prediction changed: {frame['frame_id']}")
        with np.load(path, allow_pickle=False) as arrays:
            mask, raw, probability = arrays["mask"], arrays["raw_mask"], arrays["probability"]
        if (mask.shape != frame["mask"].shape or not np.isfinite(probability).all()
                or probability.min() < 0 or probability.max() > 1):
            raise ValueError("Invalid prediction grid or probabilities")
        clean, flags = clean_mask(raw)
        if not np.array_equal(raw, probability > config["prediction_threshold"]) or not np.array_equal(mask, clean):
            raise ValueError("Frozen inference threshold or cleanup differs")
        predicted[key] = dict(frame, mask=mask)
        error = ""
        try:
            measured[key] = measure_mask(mask, frame["geometry"]["spacing_xy_mm"])
        except GeometryError as exc:
            measured[key], error = None, str(exc)
        metrics = mask_metrics(mask, frame["mask"], frame["geometry"]["spacing_xy_mm"])
        observed_frames.append({**attributes, "frame_id": frame["frame_id"], "view_phase": key,
                                "view_quality": patient[f"quality_{frame['view'].lower()}"],
                                **metrics, **flags, "one_minus_dice": 1-metrics["dice"],
                                "geometry_status": "ok" if measured[key] else "invalid_geometry", "geometry_error": error,
                                **geometry_differences(measured[key], reference[key])})
    results, observed_patients = all_methods(predicted, measured), []
    for method in METHODS:
        base, estimate = baseline[method], results[method]
        row = {**attributes, "method": method, "status": estimate["status"],
               "error": estimate.get("error", ""), "warnings": ";".join(estimate["warnings"]),
               "camus_reference_ef_pct": float(patient["reference_ef_pct"]),
               "mean_frame_dice": float(np.mean([r["dice"] for r in observed_frames]))}
        for field, unit in (("edv_ml", "edv_ml"), ("esv_ml", "esv_ml"), ("ef_pct", "ef_pp")):
            row["gt_"+field], row["predicted_"+field] = base[field], estimate[field]
            error_name = {"edv_ml": "edv_error_vs_gt_ml", "esv_ml": "esv_error_vs_gt_ml", "ef_pp": "ef_error_vs_gt_pp"}[unit]
            row[error_name] = estimate[field]-base[field] if estimate[field] is not None else None
        row["ef_error_vs_camus_pp"] = estimate["ef_pct"]-row["camus_reference_ef_pct"] if estimate["ef_pct"] is not None else None
        row.update({key: None for key in ("edv_contribution_pp", "esv_contribution_pp", "ef_rounding_remainder_pp")})
        if estimate["ef_pct"] is not None:
            row.update(volume_error_contributions(base["edv_ml"], base["esv_ml"], estimate["edv_ml"],
                                                   estimate["esv_ml"], row["ef_error_vs_gt_pp"]))
        observed_patients.append(row)
    return simulated, frame_audits, observed_frames, observed_patients


def run(prepared, predictions, output, config_path, workers=8):
    prepared, predictions, output, config_path = map(Path, (prepared, predictions, output, config_path))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty; use a new experiment directory")
    config = json.loads(config_path.read_text())
    matrix = scenarios(config)
    verify_inputs(prepared)
    dataset = CamusDataset(prepared, "test")
    ids = sorted(dataset.patient_metadata)
    if len(ids) != 50 or len(dataset.frames) != 200:
        raise ValueError("The fixed test experiment requires 50 patients and 200 frames")
    pm = json.loads((predictions/"run_metadata.json").read_text())
    if (pm["status"] != "complete" or pm["split"] != "test" or sorted(pm["patients"]) != ids
            or pm["prepared_metadata_sha256"] != sha256(prepared/"run_metadata.json")
            or pm["prediction_manifest_sha256"] != sha256(predictions/"predictions.csv")):
        raise ValueError("Prediction provenance differs from the frozen population")
    published = json.loads((ROOT/"results/prediction_metadata.json").read_text())
    if pm["checkpoint_sha256"] != published["checkpoint_sha256"]:
        raise ValueError("The experiment requires the selected published checkpoint")
    actual = index_predictions(dataset.frames, read_csv(predictions/"predictions.csv"))
    expected = read_csv(ROOT/"results/prediction_manifest.csv")
    if any(actual[r["frame_id"]]["sha256"] != r["sha256"] for r in expected):
        raise ValueError("Predictions differ from the frozen publication")
    tasks = [(str(prepared.resolve()), str(predictions.resolve()), p, config) for p in ids]
    simulations, audits, frames, patients = [], [], [], []
    print(f"Running {len(ids)} patients x {len(matrix)} scenarios x {len(METHODS)} methods", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(patient_experiment, tasks), 1):
            for target, values in zip((simulations, audits, frames, patients), result):
                target.extend(values)
            if i % 5 == 0:
                print(f"Completed {i}/{len(ids)} patients", flush=True)
    print("Computing paired uncertainty and observed-error associations", flush=True)
    summaries = paired_summary(simulations, config)
    contrasts = method_contrasts(simulations, config)
    comparisons, associations = observed_summaries(frames, patients, config)
    output.mkdir(parents=True)
    tables = {"scenarios.csv": [s.record() for s in matrix], "patients.csv": simulations,
              "perturbation_frames.csv": audits, "sensitivity_summary.csv": summaries,
              "method_contrasts.csv": contrasts, "observed_frames.csv": frames,
              "observed_patients.csv": patients, "observed_comparison.csv": comparisons,
              "associations.csv": associations}
    for filename, rows in tables.items():
        write_rows(output/filename, rows)
    summary = {"patients": len(ids), "frames": len(frames), "scenarios_per_patient": len(matrix),
               "method_count": len(METHODS), "patient_scenario_method_rows": len(simulations),
               "frame_variant_rows": len(audits), "observed_associations": len(associations),
               "zero_controls": sum(r["scenario_id"] == "control" for r in simulations),
               "failed_geometry_rows": sum(r["status"] == "invalid_geometry" for r in simulations),
               "invalid_volume_rows": sum(r["status"] == "invalid_volumes" for r in simulations),
               "border_contact_rows": sum(r["border_contact"] for r in simulations),
               "publisher_expert_exact_matches": sum(r["method"] == "publisher_original" and r["gt_ef_pct"] == r["camus_reference_ef_pct"] for r in patients),
               "observed_overall": [r for r in comparisons if r["group_type"] == "overall"]}
    atomic_json(output/"summary.json", summary)
    atomic_json(output/"config.json", config)
    code = ("analysis/sensitivity.py", "analysis/sensitivity_statistics.py", "analysis/perturbations.py",
            "geometry/comparators.py", "geometry/publisher.py", "geometry/publisher_profiles.py", "segmentation/metrics.py")
    artifacts = [*tables, "summary.json", "config.json"]
    atomic_json(output/"run_metadata.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(config_path), "patients": ids, "data_identity": identity(prepared),
        "prepared_metadata_sha256": sha256(prepared/"run_metadata.json"),
        "prediction_manifest_sha256": sha256(predictions/"predictions.csv"),
        "prediction_metadata_sha256": sha256(predictions/"run_metadata.json"),
        "checkpoint_sha256": pm["checkpoint_sha256"], "publisher_notebook_sha256": NOTEBOOK_SHA256,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__},
        "code_sha256": {name: sha256(ROOT/name) for name in code},
        "artifact_sha256": {name: sha256(output/name) for name in artifacts}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensitivity_v1"))
    parser.add_argument("--config", type=Path, default=ROOT/"analysis/configs/perturbations_v1.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run(args.prepared, args.predictions, args.output, args.config, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != "observed_overall"}, indent=2))
