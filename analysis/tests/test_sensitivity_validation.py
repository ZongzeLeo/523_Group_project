"""The independent verifier must detect numeric corruption beyond file hashes."""
import csv
import json
from pathlib import Path
import shutil

import pytest

from analysis.verify import digest
from analysis.verify_sensitivity import verify

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("defect", ["delta", "duplicate", "correlation"])
def test_corrupt_tables_are_rejected_even_if_rehashed(tmp_path, defect):
    source = ROOT/"results/sensitivity_v1"
    for path in source.glob("*.csv"):
        shutil.copyfile(path, tmp_path/path.name)
    for name in ("run_metadata.json", "config.json", "summary.json"):
        shutil.copyfile(source/name, tmp_path/name)
    name = "associations.csv" if defect == "correlation" else "patients.csv"
    with (tmp_path/name).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), reader.fieldnames
    if defect == "delta":
        rows[0]["delta_ef_pp"] = "999"
    elif defect == "duplicate":
        rows[-1] = rows[0].copy()
    else:
        rows[0]["spearman_rho"] = "0.999999"
    with (tmp_path/name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata = json.loads((tmp_path/"run_metadata.json").read_text())
    metadata["artifact_sha256"][name] = digest(tmp_path/name)
    (tmp_path/"run_metadata.json").write_bytes(json.dumps(metadata).encode())
    with pytest.raises(ValueError, match="Mismatch|Duplicate"):
        verify(ROOT, tmp_path)
