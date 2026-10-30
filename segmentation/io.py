"""Portable cached-frame I/O for Python 3.10+ GPU environments.

Reads the frozen data-stage products without changing that stage or its hashes.
Inference can request images without loading expert target arrays.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PreparedFrames:
    def __init__(self, root, split, *, load_targets=True):
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown patient split")
        self.root, self.split, self.load_targets = Path(root), split, load_targets
        self.metadata = json.loads((self.root / "run_metadata.json").read_text())
        if self.metadata["schema_version"] != 1:
            raise ValueError("Unsupported prepared schema")
        for relative, digest in self.metadata["artifact_sha256"].items():
            if sha256(self.root / relative) != digest:
                raise ValueError(f"Prepared artifact changed: {relative}")
        audit = json.loads((self.root / "audit.json").read_text())
        if audit["status"] != "passed" or audit["partial_population"]:
            raise ValueError("A complete, audited prepared dataset is required")
        with (self.root / "frames.csv").open(newline="", encoding="utf-8") as stream:
            self.frames = [row for row in csv.DictReader(stream) if row["split"] == split]
        if not self.frames:
            raise ValueError("No frames selected")
        self.patient_metadata = {row["patient_id"]: {"split": split} for row in self.frames}

    def __len__(self):
        return len(self.frames)

    def transform(self, index):
        return json.loads(self.frames[index]["transform_json"])

    def __getitem__(self, index):
        row = self.frames[index]
        path = self.root / row["prepared_path"]
        if sha256(path) != row["prepared_sha256"]:
            raise ValueError(f"Prepared frame changed: {row['frame_id']}")
        with np.load(path, allow_pickle=False) as arrays:
            image, valid = arrays["image"].copy(), arrays["valid_region"].copy()
            mask = arrays["mask"].copy() if self.load_targets else None
        shape = tuple(self.transform(index)["output_shape_yx"])
        if image.shape != shape or valid.shape != shape or not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Invalid cached image or valid-region grid")
        result = {"image": image[None], "valid_region": valid, "frame_id": row["frame_id"], "patient_id": row["patient_id"]}
        if self.load_targets:
            if mask.shape != shape or not np.isin(mask, [0, 1]).all():
                raise ValueError("Invalid cached binary target")
            result["mask"] = mask
        return result
