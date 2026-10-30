"""Download the official CAMUS public ED/ES release, with resumable receipts.

Uses Python's standard library. Cine sequences are intentionally not required
for this project's annotated ED/ES workflow. Source files are never converted.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
COLLECTION = "6373703d73e9f0047faa1bc8"
FOLDERS = {
    "database_nifti": "63fde55f73e9f004868fb7ac",
    "database_split": "66e27d12961576b1bad4e4e1",
    "jupyter": "64b5a9b473e9f00492ce9036",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def fetch(endpoint: str) -> bytes:
    """Retry transient network failures; never disable TLS verification."""
    for attempt in range(4):
        try:
            request = Request(API + endpoint, headers={"User-Agent": "ELEC523-CAMUS-preparation/1"})
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def listing(resource: str, **params) -> list[dict]:
    result = []
    for offset in range(0, 100000, 100):
        page = json.loads(fetch("/" + resource + "?" + urlencode(
            dict(params, limit=100, offset=offset, sort="name", sortdir=1))))
        result.extend(page)
        if len(page) < 100:
            return result
    raise ValueError("Unexpectedly large API listing")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def download_items(root: Path, relative_dir: Path, items: list[dict], receipt: Path) -> list[dict]:
    previous = {}
    if receipt.exists():
        previous = {entry["path"]: entry for entry in json.loads(receipt.read_text(encoding="utf-8"))}
    records = []
    for item in items:
        name = item["name"]
        if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid remote filename: {name!r}")
        relative_path = (relative_dir / name).as_posix()
        destination = root / relative_path
        old = previous.get(relative_path, {})
        if (destination.is_file() and old.get("item_id") == item["_id"]
                and destination.stat().st_size == item["size"]
                and old.get("sha256") == sha256(destination)):
            records.append(old)
            continue
        endpoint = f"/item/{item['_id']}/download"
        content = fetch(endpoint)
        if len(content) != item["size"]:
            raise ValueError(f"Download size mismatch: {relative_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(destination)
        records.append({
            "path": relative_path, "item_id": item["_id"], "url": API + endpoint,
            "bytes": len(content), "sha256": sha256(destination),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        })
        # Each completed file is durable, even if a later request fails.
        atomic_json(receipt, records)
    atomic_json(receipt, records)
    return records


def download(output: Path, workers: int = 6, patients: list[str] | None = None) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(fetch(f"/collection/{COLLECTION}"))
    if metadata.get("name") != "CAMUS_public" or not metadata.get("public"):
        raise ValueError("Unexpected CAMUS collection metadata")
    records = []
    for name, folder_id in FOLDERS.items():
        records.extend(download_items(output, Path(name), listing("item", folderId=folder_id),
                                      output / ".receipts" / f"{name}.json"))
    folders = listing("folder", parentType="folder", parentId=FOLDERS["database_nifti"])
    folders = [folder for folder in folders if re.fullmatch(r"patient\d{4}", folder["name"])]
    if patients:
        missing = set(patients) - {folder["name"] for folder in folders}
        if missing:
            raise ValueError(f"Unknown patient IDs: {sorted(missing)}")
        folders = [folder for folder in folders if folder["name"] in patients]
    elif len(folders) != 500:
        raise ValueError(f"Expected 500 official patient folders, found {len(folders)}")

    def patient_download(folder):
        patient = folder["name"]
        items = listing("item", folderId=folder["_id"])
        expected = {f"{patient}_{view}_{phase}{suffix}.nii.gz"
                    for view in ("2CH", "4CH") for phase in ("ED", "ES") for suffix in ("", "_gt")}
        expected.update({"Info_2CH.cfg", "Info_4CH.cfg", "MANDATORY_CITATION.md"})
        selected = [item for item in items if item["name"] in expected]
        if {item["name"] for item in selected} != expected or len(selected) != len(expected):
            raise ValueError(f"Missing or duplicate source items for {patient}")
        return download_items(output, Path("database_nifti") / patient, selected,
                              output / ".receipts" / f"{patient}.json")

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(patient_download, folder): folder["name"] for folder in folders}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                records.extend(future.result())
            except Exception as exc:
                errors.append({"patient_id": futures[future], "error": str(exc)})
            if number % 25 == 0 or number == len(folders):
                print(f"Downloaded/verified {number}/{len(folders)} patients; failures={len(errors)}", flush=True)
    manifest = {
        "collection_id": COLLECTION, "api": API, "collection_name": metadata["name"],
        "scope": "annotated ED/ES images, masks, CFGs, published splits and source documentation",
        "patient_count_requested": len(folders), "complete_500_patient_download": len(folders) == 500 and not errors,
        "files": sorted(records, key=lambda record: record["path"]), "errors": errors,
        "checksum_note": "Local SHA-256 and server byte counts; no publisher checksum was supplied.",
    }
    atomic_json(output / "download_manifest.json", manifest)
    if errors:
        raise RuntimeError(f"{len(errors)} patient downloads failed; rerun to resume. See download_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--patients", nargs="+", help="Optional explicit subset for a smoke test")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    manifest = download(args.output, args.workers, args.patients)
    print(f"Verified {len(manifest['files'])} source files in {args.output.resolve()}")


if __name__ == "__main__":
    main()
