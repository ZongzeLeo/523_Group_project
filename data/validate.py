"""Read back every prepared frame and save an inspectable validation receipt."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np

from .dataset import CamusDataset
from .download_camus import atomic_json, sha256
from .prepare import prepare, verify_prepared


def validate(prepared_root, report_path, *, check_reuse=False):
    root = Path(prepared_root).resolve()
    metadata = verify_prepared(root)
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    recipe = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    counts, patient_sets, native_checks = {}, {}, []
    for split in ("train", "validation", "test"):
        dataset = CamusDataset(root, split)
        patient_sets[split] = set(dataset.patient_metadata)
        for sample in dataset:
            if (sample["image"].dtype != np.float32 or sample["mask"].dtype != np.uint8
                    or sample["valid_region"].dtype != bool or not sample["mask"].any()
                    or not np.all(sample["image"][0][~sample["valid_region"]] == 0)
                    or not np.all(sample["mask"][~sample["valid_region"]] == 0)):
                raise ValueError(f"Array contract failed: {sample['frame_id']}")
        counts[split] = {"patients": len(dataset.patient_metadata), "frames": len(dataset)}
        patient_id = next(iter(dataset.patient_metadata))
        paired = dataset.patient_frames(patient_id, native=True)
        for item in paired.values():
            np.testing.assert_array_equal(item["mask"], item["multiclass_mask"] == 1)
            native_checks.append(item["frame_id"])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if patient_sets[left] & patient_sets[right]:
            raise ValueError(f"Patient leakage between {left} and {right}")
    torch_check = {"status": "not_installed_optional"}
    if importlib.util.find_spec("torch"):
        import torch
        from torch.utils.data import DataLoader
        batch = next(iter(DataLoader(CamusDataset(root, "train"), batch_size=8, num_workers=0)))
        if list(batch["image"].shape) != [8, 1, *recipe["output_shape_yx"]]:
            raise ValueError("Unexpected PyTorch batch shape")
        torch_check = {"status": "passed", "version": torch.__version__,
                       "image_shape": list(batch["image"].shape), "mask_shape": list(batch["mask"].shape)}
    reuse = "not_requested"
    if check_reuse:
        before = sha256(root / "run_metadata.json")
        raw = (root / metadata["raw_root_relative_to_prepared"]).resolve()
        prepare(raw, root, tuple(recipe["output_shape_yx"]), recipe["expected_patients"], recipe["allow_partial"])
        if before != sha256(root / "run_metadata.json"):
            raise ValueError("Frozen rerun changed metadata")
        reuse = "passed: source hashes, recipe and code matched; all outputs verified and unchanged"
    raw = (root / metadata["raw_root_relative_to_prepared"]).resolve()
    acquisition = {}
    if (raw / "download_manifest.json").exists():
        download = json.loads((raw / "download_manifest.json").read_text(encoding="utf-8"))
        acquisition = {"collection_id": download["collection_id"], "source_file_count": len(download["files"]),
                       "source_bytes": sum(row["bytes"] for row in download["files"]),
                       "download_manifest_sha256": sha256(raw / "download_manifest.json"),
                       "complete_500_patient_download": download["complete_500_patient_download"]}
    receipt = {
        "status": "passed", "validated_utc": datetime.now(timezone.utc).isoformat(),
        "run_metadata_sha256": sha256(root / "run_metadata.json"),
        "input_fingerprint": metadata["input_fingerprint"], "acquisition": acquisition,
        "counts": counts, "all_prepared_frames_read": sum(value["frames"] for value in counts.values()),
        "all_artifact_and_array_hashes_verified": True, "patient_overlap_between_splits": 0,
        "cross_split_duplicate_image_groups": audit["cross_split_duplicate_image_groups"],
        "native_pair_readback_frames": native_checks, "torch_dataloader": torch_check,
        "frozen_rerun": reuse, "data_warnings": audit["warnings"],
        "reference_missingness": {key: audit[key] for key in ("missing_reference_ef", "missing_reference_edv", "missing_reference_esv")},
        "scope": "Data preparation only; no model training, volume estimation, or prediction-performance validation.",
    }
    atomic_json(Path(report_path), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/validation.json"))
    parser.add_argument("--check-reuse", action="store_true")
    args = parser.parse_args()
    result = validate(args.prepared_root, args.report, check_reuse=args.check_reuse)
    print(json.dumps({key: result[key] for key in ("status", "counts", "torch_dataloader", "frozen_rerun")}, indent=2))
