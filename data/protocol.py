"""Content-based dataset identity and native-source integrity checks."""
import csv
import hashlib
import json
from pathlib import Path

from .download_camus import sha256


def identity(prepared):
    """Ignore timestamps and newline conventions; retain source, split and recipe."""
    root = Path(prepared)
    values = {name: json.loads((root / name).read_text(encoding="utf-8"))
              for name in ("source_hashes.json", "preprocessing.json")}
    with (root / "splits.csv").open(encoding="utf-8", newline="") as stream:
        values["splits.csv"] = sorted(csv.DictReader(stream), key=lambda row: row["patient_id"])
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for name, value in values.items()}


def verify_inputs(prepared_root):
    root = Path(prepared_root)
    metadata = verify_results(root)
    raw = (root / metadata["raw_root_relative_to_prepared"]).resolve()
    sources = json.loads((root / "source_hashes.json").read_text(encoding="utf-8"))
    for relative, digest in sources.items():
        if sha256(raw / relative) != digest:
            raise ValueError(f"Native source changed: {relative}")
    return metadata


def verify_results(output):
    root = Path(output)
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    for relative, digest in metadata["artifact_sha256"].items():
        if sha256(root / relative) != digest:
            raise ValueError(f"Artifact changed: {relative}")
    return metadata
