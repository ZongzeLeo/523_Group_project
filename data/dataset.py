"""Small NumPy dataset compatible with torch.utils.data.DataLoader.

Reference measurements remain in patient_metadata; they are not model inputs.
Native expert masks and geometry remain available for the subsequent volume step.
"""

import csv
import json
from pathlib import Path

import numpy as np

from .download_camus import sha256
from .native import read_pair
from .prepare import SPLITS, verify_prepared


class CamusDataset:
    def __init__(self, prepared_root="data/prepared", split="train", *, raw_root=None,
                 allow_partial=False, verify=False):
        self.root = Path(prepared_root).resolve()
        if split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}")
        metadata = json.loads((self.root / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1:
            raise ValueError("Unsupported prepared dataset schema")
        for filename in ("frames.csv", "patients.csv", "splits.csv", "audit.json"):
            if sha256(self.root / filename) != metadata["artifact_sha256"][filename]:
                raise ValueError(f"Frozen manifest changed: {filename}")
        audit = json.loads((self.root / "audit.json").read_text(encoding="utf-8"))
        if audit["status"] != "passed" or (audit["partial_population"] and not allow_partial):
            raise ValueError("Preparation failed or contains only a partial population")
        if verify:
            verify_prepared(self.root)
        self.raw_root = Path(raw_root).resolve() if raw_root else (self.root / metadata["raw_root_relative_to_prepared"]).resolve()
        with (self.root / "frames.csv").open(newline="", encoding="utf-8") as stream:
            self.frames = [row for row in csv.DictReader(stream) if row["split"] == split]
        with (self.root / "patients.csv").open(newline="", encoding="utf-8") as stream:
            self.patient_metadata = {row["patient_id"]: row for row in csv.DictReader(stream) if row["split"] == split}
        self.split = split
        if not self.frames:
            raise ValueError(f"No frames in split {split}")

    def __len__(self):
        return len(self.frames)

    def transform(self, index):
        return json.loads(self.frames[index]["transform_json"])

    def __getitem__(self, index):
        row = self.frames[index]
        with np.load(self.root / row["prepared_path"], allow_pickle=False) as arrays:
            image, mask, valid = arrays["image"].copy(), arrays["mask"].copy(), arrays["valid_region"].copy()
        transform = self.transform(index)
        expected_shape = tuple(transform["output_shape_yx"])
        if (image.shape != expected_shape or mask.shape != expected_shape or valid.shape != expected_shape
                or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1
                or not np.isin(mask, [0, 1]).all()):
            raise ValueError(f"Invalid prepared array: {row['frame_id']}")
        return {
            "image": image[np.newaxis, ...], "mask": mask, "valid_region": valid,
            "patient_id": row["patient_id"], "frame_id": row["frame_id"], "view": row["view"], "phase": row["phase"],
            "model_to_native_xy": np.asarray(transform["model_to_native_xy"], dtype=np.float64),
            "native_shape_yx": np.asarray(transform["native_shape_yx"], dtype=np.int64),
            "native_spacing_xy_mm": np.asarray(transform["native_spacing_xy_mm"], dtype=np.float64),
        }

    def native(self, index):
        row = self.frames[index]
        image_path, mask_path = self.raw_root / row["image_path"], self.raw_root / row["mask_path"]
        if sha256(image_path) != row["image_sha256"] or sha256(mask_path) != row["mask_sha256"]:
            raise ValueError(f"Native source changed: {row['frame_id']}")
        image, labels, geometry = read_pair(image_path, mask_path)
        return {"image": image, "mask": (labels == 1).astype(np.uint8), "multiclass_mask": labels,
                "geometry": geometry, "patient_id": row["patient_id"], "frame_id": row["frame_id"],
                "view": row["view"], "phase": row["phase"]}

    def patient_frames(self, patient_id, *, native=True):
        if patient_id not in self.patient_metadata:
            raise KeyError(f"Patient {patient_id} is not in the {self.split} split")
        result = {f"{row['view']}_{row['phase']}": self.native(index) if native else self[index]
                  for index, row in enumerate(self.frames) if row["patient_id"] == patient_id}
        if len(result) != 4:
            raise ValueError(f"Incomplete four-frame pair: {patient_id}")
        return result
