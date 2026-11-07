"""Download the public CAMUS release (ED/ES frames, masks, Info cfgs, official split).

Source: CREATIS Human Heart Project data server (Girder API, no login needed).
Skips the half-sequence cine files, which this project doesn't use.
Re-running resumes: files already present with the right size are skipped.

Usage (run on a node with internet access, e.g. the login node):
    python download_camus.py --out /data/scratch/users/<you>/CAMUS
Then set DATA_ROOT=<out>/database_nifti in train_and_eval.sbatch.

Please cite Leclerc et al., IEEE TMI 2019 when using CAMUS.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
NIFTI_FOLDER = "63fde55f73e9f004868fb7ac"   # database_nifti
SPLIT_FOLDER = "66e27d12961576b1bad4e4e1"   # database_split


def get(endpoint, params=None, retries=4):
    url = f"{API}/{endpoint}" + (f"?{urlencode(params)}" if params else "")
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "camus-downloader"}), timeout=120) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def listing(kind, **params):
    out, offset = [], 0
    while True:
        page = json.loads(get(kind, {**params, "limit": 500, "offset": offset, "sort": "name"}))
        out += page
        if len(page) < 500:
            return out
        offset += 500


def wanted(name):
    """ED/ES images + masks and the Info cfgs; skip cine half-sequences."""
    return name.endswith(".cfg") or (name.endswith(".nii.gz") and "half_sequence" not in name)


def download_item(item, dest: Path):
    if dest.exists() and dest.stat().st_size == item["size"]:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(get(f"item/{item['_id']}/download"))
    if tmp.stat().st_size != item["size"]:
        tmp.unlink()
        raise IOError(f"size mismatch for {dest.name}")
    tmp.rename(dest)
    return "downloaded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder (gets database_nifti/ and database_split/)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patients", type=int, default=None, help="only the first N patients (quick test)")
    args = ap.parse_args()
    out = Path(args.out)

    jobs = [(it, out / "database_split" / it["name"]) for it in listing("item", folderId=SPLIT_FOLDER)]
    patients = [f for f in listing("folder", parentType="folder", parentId=NIFTI_FOLDER)
                if f["name"].startswith("patient")]
    if args.patients:
        patients = patients[:args.patients]
    print(f"listing files for {len(patients)} patients...", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        for folder, items in zip(patients, ex.map(lambda f: listing("item", folderId=f["_id"]), patients)):
            jobs += [(it, out / "database_nifti" / folder["name"] / it["name"])
                     for it in items if wanted(it["name"])]

    print(f"{len(jobs)} files to fetch", flush=True)
    counts, failed = {"downloaded": 0, "skipped": 0}, []
    with ThreadPoolExecutor(args.workers) as ex:
        futures = {ex.submit(download_item, it, dest): dest for it, dest in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as e:
                failed.append((futures[fut], e))
            if n % 200 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} done ({counts}, failed {len(failed)})", flush=True)

    for dest, e in failed[:10]:
        print(f"FAILED {dest}: {e}")
    if failed:
        raise SystemExit(f"{len(failed)} files failed; re-run the same command to resume")
    n_pat = sum(1 for p in (out / "database_nifti").iterdir() if p.is_dir())
    print(f"done: {n_pat} patient folders in {out / 'database_nifti'}")


if __name__ == "__main__":
    main()