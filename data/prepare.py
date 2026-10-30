"""Inventory and audit CAMUS, then build frozen manifests and training arrays."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re

import nibabel as nib
import numpy as np
import scipy
import SimpleITK as sitk

from .download_camus import atomic_json, sha256
from .native import read_config, read_pair
from .transforms import preprocess

VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")
SPLITS = {"train": "subgroup_training.txt", "validation": "subgroup_validation.txt", "test": "subgroup_testing.txt"}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_splits(split_dir: Path) -> tuple[dict, dict]:
    assignments, hashes = {}, {}
    for split, filename in SPLITS.items():
        path = split_dir / filename
        hashes[filename] = sha256(path)
        for patient in path.read_text(encoding="utf-8-sig").split():
            if not re.fullmatch(r"patient\d{4}", patient):
                raise ValueError(f"Invalid split patient ID: {patient!r}")
            if patient in assignments:
                raise ValueError(f"Duplicate patient/split leakage: {patient}")
            assignments[patient] = split
    if not assignments or set(assignments.values()) != set(SPLITS):
        raise ValueError("All three published split lists must be nonempty")
    return assignments, hashes


def ef_group(value):
    if value is None:
        return "missing"
    return "le45" if value <= 45 else "gt45_lt55" if value < 55 else "ge55"


def numeric(cfg, key, *, required=False):
    text = cfg.get(key, "").strip()
    if not text:
        if required:
            raise ValueError(f"Missing CFG field: {key}")
        return None
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite CFG field: {key}")
    return value


def inventory(raw: Path, expected_patients: int, allow_partial: bool):
    assignments, split_hashes = read_splits(raw / "database_split")
    directories = sorted(path for path in (raw / "database_nifti").glob("patient*") if path.is_dir())
    found = {path.name for path in directories}
    errors, warnings, frames, patients = [], [], [], []
    if expected_patients < 1:
        raise ValueError("expected_patients must be positive")
    if len(assignments) != expected_patients:
        errors.append({"code": "split_population", "detail": f"Expected {expected_patients}, split lists contain {len(assignments)}"})
    if not allow_partial and found != set(assignments):
        errors.append({"code": "incomplete_population", "missing": sorted(set(assignments) - found), "extra": sorted(found - set(assignments))})
    if not found or found - set(assignments):
        errors.append({"code": "unassigned_or_empty_population", "extra": sorted(found - set(assignments))})
    receipts = {}
    receipt_path = raw / "download_manifest.json"
    if receipt_path.exists():
        receipts = {row["path"]: row for row in json.loads(receipt_path.read_text(encoding="utf-8"))["files"]}
    source_hashes = {}

    def source_hash(path):
        relative = path.relative_to(raw).as_posix()
        digest = sha256(path)
        if receipts and (relative not in receipts or receipts[relative]["sha256"] != digest):
            raise ValueError(f"Source differs from download receipt: {relative}")
        source_hashes[relative] = digest
        return digest

    for filename in SPLITS.values():
        source_hash(raw / "database_split" / filename)
    pixels_seen = defaultdict(list)
    for directory in directories:
        patient_id = directory.name
        if patient_id not in assignments:
            continue
        config = {}
        for view in VIEWS:
            config_path = directory / f"Info_{view}.cfg"
            try:
                source_hash(config_path)
                cfg = read_config(config_path)
                ef = numeric(cfg, "EF")
                if ef is not None and not 0 <= ef <= 100:
                    raise ValueError("Reference EF must be in [0, 100] percent")
                n_frames = numeric(cfg, "NbFrame", required=True)
                for phase in PHASES:
                    index = numeric(cfg, phase, required=True)
                    if index != int(index) or not 1 <= index <= n_frames:
                        raise ValueError("Invalid advisory phase index")
                if n_frames != int(n_frames) or cfg["ED"] == cfg["ES"]:
                    raise ValueError("Invalid frame count or equal ED/ES indices")
                config[view] = (cfg, ef)
                if cfg.get("ImageQuality") not in {"Good", "Medium", "Poor"}:
                    warnings.append({"code": "missing_or_unknown_quality", "patient_id": patient_id, "view": view})
            except Exception as exc:
                errors.append({"code": "config", "patient_id": patient_id, "view": view, "detail": str(exc)})
                continue
            for phase in PHASES:
                frame_id = f"{patient_id}_{view}_{phase}"
                image_path = directory / f"{frame_id}.nii.gz"
                mask_path = directory / f"{frame_id}_gt.nii.gz"
                try:
                    image_hash, mask_hash = source_hash(image_path), source_hash(mask_path)
                    image, labels, geometry = read_pair(image_path, mask_path)
                    pixel_hash = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
                    pixels_seen[pixel_hash].append({"patient_id": patient_id, "frame_id": frame_id,
                                                    "split": assignments[patient_id]})
                    frames.append({
                        "frame_id": frame_id, "patient_id": patient_id, "split": assignments[patient_id],
                        "view": view, "phase": phase, "phase_source": "named_ED_ES_file",
                        "cfg_frame_index_advisory": cfg[phase], "image_quality": cfg.get("ImageQuality", ""),
                        "image_path": image_path.relative_to(raw).as_posix(),
                        "mask_path": mask_path.relative_to(raw).as_posix(),
                        "config_path": config_path.relative_to(raw).as_posix(),
                        "native_height": image.shape[0], "native_width": image.shape[1],
                        "spacing_x_mm": geometry["spacing_xy_mm"][0], "spacing_y_mm": geometry["spacing_xy_mm"][1],
                        "image_min": float(image.min()), "image_max": float(image.max()),
                        "lv_pixels": int((labels == 1).sum()), "labels": json.dumps(geometry["labels"]),
                        "native_geometry_json": json.dumps(geometry, sort_keys=True, separators=(",", ":")),
                        "image_sha256": image_hash, "mask_sha256": mask_hash, "image_pixels_sha256": pixel_hash,
                    })
                except Exception as exc:
                    errors.append({"code": "frame", "frame_id": frame_id, "detail": str(exc)})
        if len(config) != 2:
            continue
        ef2, ef4 = config["2CH"][1], config["4CH"][1]
        ef = ef2 if ef2 is not None and ef2 == ef4 else None
        if ef is None:
            warnings.append({"code": "missing_or_conflicting_reference_ef", "patient_id": patient_id, "ef_2ch": ef2, "ef_4ch": ef4})
        patients.append({
            "patient_id": patient_id, "split": assignments[patient_id],
            **{f"frame_{view}_{phase}": f"{patient_id}_{view}_{phase}" for view in VIEWS for phase in PHASES},
            "reference_ef_pct": ef, "reference_ef_group": ef_group(ef),
            "reference_ef_2ch_pct": ef2, "reference_ef_4ch_pct": ef4,
            "reference_ef_source": "Info_2CH.cfg and Info_4CH.cfg" if ef is not None else "missing_or_conflicting",
            "reference_edv_ml": None, "reference_esv_ml": None,
            "reference_volume_status": "not_supplied_in_CFG",
            "quality_2ch": config["2CH"][0].get("ImageQuality", ""),
            "quality_4ch": config["4CH"][0].get("ImageQuality", ""),
        })
    duplicate_groups = []
    for pixel_hash, entries in pixels_seen.items():
        if len({entry["patient_id"] for entry in entries}) > 1:
            cross_split = len({entry["split"] for entry in entries}) > 1
            finding = {"code": "duplicate_pixels_across_splits" if cross_split else "duplicate_pixels_within_split",
                       "image_pixels_sha256": pixel_hash, "frames": entries,
                       "detail": "Identical image pixels under distinct patient IDs; retained only if confined to one split."}
            duplicate_groups.append(finding)
            (errors if cross_split else warnings).append(finding)
    if len(frames) != 4 * len(directories):
        errors.append({"code": "incomplete_view_phase_pairs", "expected_frames": 4 * len(directories), "actual_frames": len(frames)})
    counts = {split: sum(row["split"] == split for row in patients) for split in SPLITS}
    report = {
        "status": "failed" if errors else "passed", "patient_count": len(patients), "frame_count": len(frames),
        "expected_patients": expected_patients, "partial_population": found != set(assignments),
        "patient_counts_by_split": counts,
        "frame_counts_by_split": dict(Counter(row["split"] for row in frames)),
        "patient_overlap_between_splits": 0,
        "cross_patient_duplicate_image_groups": duplicate_groups,
        "cross_split_duplicate_image_groups": sum(group["code"] == "duplicate_pixels_across_splits" for group in duplicate_groups),
        "missing_reference_ef": sum(row["reference_ef_pct"] is None for row in patients),
        "missing_reference_edv": len(patients), "missing_reference_esv": len(patients),
        "reference_volume_note": "No standalone EDV/ESV scalar reference fields in the CFG source; geometry is a later stage.",
        "ef_groups_by_split": {split: dict(Counter(row["reference_ef_group"] for row in patients if row["split"] == split)) for split in SPLITS},
        "image_quality_by_split": {split: dict(Counter(row["image_quality"] for row in frames if row["split"] == split)) for split in SPLITS},
        "shape_yx_counts": dict(Counter(f"{row['native_height']}x{row['native_width']}" for row in frames)),
        "spacing_xy_mm_values": sorted({(row["spacing_x_mm"], row["spacing_y_mm"]) for row in frames}),
        "checks": ["patient uniqueness and split disjointness", "four image/mask pairs per patient", "source SHA-256 receipts when supplied", "image/mask size, origin, direction and spacing agreement with explicit mm units", "finite intensity range and nonconstant images", "integer mask labels in 0..3 and nonempty LV label 1", "duplicate pixel content across patients, blocking if across splits", "CFG phase-index ranges and EF consistency"],
        "limitations": ["Named ED/ES snapshots are used directly; CFG indices are advisory and not cross-checked against cine sequences.", "Pixel-content duplicate detection does not establish independence beyond the publisher's patient identifiers.", "All image-quality groups are retained; quality is recorded for downstream stratification."],
        "errors": errors, "warnings": warnings,
    }
    return frames, patients, report, source_hashes, split_hashes


def verify_prepared(output: Path) -> dict:
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    for relative, expected in metadata["artifact_sha256"].items():
        if sha256(output / relative) != expected:
            raise ValueError(f"Prepared artifact changed: {relative}")
    with (output / "frames.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if sha256(output / row["prepared_path"]) != row["prepared_sha256"]:
                raise ValueError(f"Prepared array changed: {row['frame_id']}")
    return metadata


def prepare(raw: Path, output: Path, output_shape=(256, 256), expected_patients=500, allow_partial=False):
    raw, output = raw.resolve(), output.resolve()
    if len(output_shape) != 2 or min(output_shape) < 2:
        raise ValueError("Output shape must have two dimensions >= 2")
    if output == raw or output in raw.parents or raw in output.parents:
        raise ValueError("Raw and prepared roots must be separate, non-nested directories")
    try:
        frames, patients, report, sources, split_hashes = inventory(raw, expected_patients, allow_partial)
    except (ValueError, OSError, KeyError) as exc:
        atomic_json(output.with_name(output.name + ".failed_audit.json"), {
            "status": "failed", "errors": [{"code": "source_or_split_inventory", "detail": str(exc)}]})
        raise
    if report["errors"]:
        atomic_json(output.with_name(output.name + ".failed_audit.json"), report)
        raise ValueError(f"CAMUS audit failed with {len(report['errors'])} errors; see {output.name}.failed_audit.json")
    recipe = {"schema_version": 1, "output_shape_yx": list(output_shape), "normalization": "fixed image / 255",
              "resampling": "isotropic physical letterbox; linear images, nearest LV labels", "lv_label": 1,
              "expected_patients": expected_patients, "allow_partial": allow_partial}
    code_hashes = {name: sha256(Path(__file__).parent / name)
                   for name in ("prepare.py", "native.py", "transforms.py", "download_camus.py")}
    fingerprint = hashlib.sha256(json.dumps({"recipe": recipe, "sources": sources, "code": code_hashes}, sort_keys=True).encode()).hexdigest()
    if (output / "run_metadata.json").exists():
        previous = verify_prepared(output)
        if previous["input_fingerprint"] != fingerprint:
            raise ValueError("Frozen preparation differs from the current source/code/recipe. Use a new --output directory.")
        print("Existing frozen preparation verified; no files changed.", flush=True)
        return report
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty and has no completion marker; use a new --output directory.")
    (output / "arrays").mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(frames, 1):
        image, labels, geometry = read_pair(raw / row["image_path"], raw / row["mask_path"])
        image, mask, valid, transform = preprocess(image, labels, geometry["spacing_xy_mm"], output_shape)
        relative = f"arrays/{row['frame_id']}.npz"
        np.savez_compressed(output / relative, image=image, mask=mask, valid_region=valid)
        row.update({"prepared_path": relative, "prepared_sha256": sha256(output / relative),
                    "transform_json": json.dumps(transform, sort_keys=True, separators=(",", ":"))})
        if index % 250 == 0 or index == len(frames):
            print(f"Prepared {index}/{len(frames)} frames", flush=True)
    write_csv(output / "frames.csv", frames)
    write_csv(output / "patients.csv", patients)
    write_csv(output / "splits.csv", [{"patient_id": row["patient_id"], "split": row["split"]} for row in patients])
    atomic_json(output / "audit.json", report)
    atomic_json(output / "source_hashes.json", sources)
    atomic_json(output / "preprocessing.json", recipe)
    artifacts = ["frames.csv", "patients.csv", "splits.csv", "audit.json", "source_hashes.json", "preprocessing.json"]
    try:
        raw_relative = Path(os.path.relpath(raw, output)).as_posix()
    except ValueError:
        raw_relative = str(raw)
    atomic_json(output / "run_metadata.json", {
        "schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root_relative_to_prepared": raw_relative, "input_fingerprint": fingerprint,
        "source_type": "official CAMUS download receipts" if (raw / "download_manifest.json").exists() else "locally supplied NIfTI files",
        "source_collection": "6373703d73e9f0047faa1bc8", "split_file_sha256": split_hashes,
        "code_sha256": code_hashes, "artifact_sha256": {name: sha256(output / name) for name in artifacts},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "SimpleITK": sitk.Version_VersionString(), "nibabel": nib.__version__},
    })
    verify_prepared(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--size", nargs=2, type=int, default=(256, 256), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--expected-patients", type=int, default=500)
    parser.add_argument("--allow-partial", action="store_true", help="Explicit smoke-test subset; recorded in audit")
    parser.add_argument("--verify-only", action="store_true", help="Verify cached manifests and all NPZ hashes")
    args = parser.parse_args()
    if args.verify_only:
        verify_prepared(args.output)
        print("Prepared artifacts and all array hashes verified.")
    else:
        report = prepare(args.raw_root, args.output, tuple(args.size), args.expected_patients, args.allow_partial)
        print(json.dumps({key: report[key] for key in ("status", "patient_count", "frame_count", "patient_counts_by_split", "missing_reference_ef", "missing_reference_edv", "missing_reference_esv")}, indent=2))


if __name__ == "__main__":
    main()
