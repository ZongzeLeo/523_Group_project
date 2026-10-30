"""Persist a direct original-notebook expert-mask reproduction on a fixed split."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import platform

import numpy as np
import PIL
import skimage

from data.dataset import CamusDataset
from data.download_camus import atomic_json, sha256
from data.prepare import write_csv
from . import GeometryError
from .publisher import measure_publisher, NOTEBOOK_SHA256
from data.protocol import verify_inputs, identity


@lru_cache(maxsize=4)
def _dataset(root, split):
    return CamusDataset(root, split)


def _patient(task):
    root, split, patient_id = task
    dataset = _dataset(root, split)
    reference = float(dataset.patient_metadata[patient_id]["reference_ef_pct"])
    try:
        result = measure_publisher(dataset.patient_frames(patient_id))
        error = result["ef_pct"] - reference
        return {"patient_id": patient_id, "split": split, **result, "reference_ef_pct": reference,
                "difference_pp": error, "exact_match": error == 0, "error": ""}
    except GeometryError as exc:
        return {"patient_id": patient_id, "split": split, "edv_ml": None, "esv_ml": None,
                "ef_pct": None, "ef_before_final_rounding_pct": None, "status": "invalid_geometry",
                "warnings": [], "reference_ef_pct": reference, "difference_pp": None,
                "exact_match": False, "error": str(exc)}


def reproduce(prepared, output, split="test", workers=4):
    prepared, output = Path(prepared), Path(output)
    verify_inputs(prepared)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory for a new reproduction")
    dataset = _dataset(str(prepared.resolve()), split)
    tasks = [(str(prepared.resolve()), split, patient) for patient in dataset.patient_metadata]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_patient, tasks))
    valid = [row for row in rows if row["status"] == "ok"]
    summary = {"split": split, "attempted_patients": len(rows), "valid_patients": len(valid),
               "failed_patients": len(rows)-len(valid), "exact_ef_matches": sum(row["exact_match"] for row in valid),
               "mae_pp": float(np.mean([abs(row["difference_pp"]) for row in valid])) if valid else None,
               "notebook_sha256": NOTEBOOK_SHA256,
               "volume_reference": "These are original-notebook GT-mask-derived volumes, not independent clinical volume labels."}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "patients.csv", [dict(row, warnings=";".join(row["warnings"])) for row in rows])
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "run_metadata.json", {
        "prepared_metadata_sha256": sha256(prepared / "run_metadata.json"),
        "data_identity": identity(prepared),
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in ("publisher.py", "reproduce_publisher.py")},
        "notebook_sha256": NOTEBOOK_SHA256,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scikit_image": skimage.__version__, "pillow": PIL.__version__},
        "artifact_sha256": {name: sha256(output / name) for name in ("patients.csv", "summary.json")}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("outputs/geometry"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.prepared, args.output, args.split, args.workers), indent=2))
