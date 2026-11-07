"""Patient-level train/val/test split for CAMUS.

Recommended: --official-split-dir <CAMUS>/database_split uses the publisher's
patient-level split (400 train / 50 val / 50 test), matching the team's other
workflow. Without it: a split stratified by EF category x worst view quality.

Outputs:
    splits.json   {"train": [...], "val": [...], "test": [...], "method": ...}
    patients.csv  per-patient metadata + split (share with teammates)
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import QUALITY_ORDER, VIEWS, patient_ids, patient_meta


def read_official(split_dir):
    names = {"train": "subgroup_training.txt", "val": "subgroup_validation.txt",
             "test": "subgroup_testing.txt"}
    missing = [f for f in names.values() if not (Path(split_dir) / f).exists()]
    if missing:
        raise FileNotFoundError(f"{split_dir} is missing {missing}; found "
                                f"{sorted(p.name for p in Path(split_dir).iterdir())}")
    return {k: [l.strip() for l in (Path(split_dir) / f).read_text().splitlines() if l.strip()]
            for k, f in names.items()}


def read_split_csv(path):
    """CSV with columns patient_id, split (train / validation|val / test),
    e.g. the team repo's data/reference/splits.csv."""
    df = pd.read_csv(path)
    df["split"] = df["split"].replace({"validation": "val"})
    bad = set(df["split"]) - {"train", "val", "test"}
    if bad:
        raise ValueError(f"unknown split names in {path}: {bad}")
    return {k: sorted(df.loc[df["split"] == k, "patient_id"]) for k in ("train", "val", "test")}


def stratified(meta, val_frac, test_frac, seed):
    strata = defaultdict(list)
    for m in meta:
        worst = max((m[f"quality_{v}"] for v in VIEWS), key=lambda q: QUALITY_ORDER.get(q, -1))
        strata[(m["ef_category"], worst)].append(m["patient"])
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for key in sorted(strata, key=str):
        pids = sorted(strata[key])
        rng.shuffle(pids)
        n = len(pids)
        n_test, n_val = round(n * test_frac), round(n * val_frac)
        splits["test"] += pids[:n_test]
        splits["val"] += pids[n_test:n_test + n_val]
        splits["train"] += pids[n_test + n_val:]
    return {k: sorted(v) for k, v in splits.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-csv", default=None,
                    help="patient_id,split CSV (e.g. team repo data/reference/splits.csv); "
                         "takes priority over the other options")
    ap.add_argument("--official-split-dir", default=None,
                    help="CAMUS database_split folder; overrides stratified split")
    args = ap.parse_args()

    meta = [patient_meta(args.data_root, pid) for pid in patient_ids(args.data_root)]
    if args.split_csv:
        splits = read_split_csv(args.split_csv)
        method = f"split list from {Path(args.split_csv).name} (official CAMUS 400/50/50)"
    elif args.official_split_dir:
        splits = read_official(args.official_split_dir)
        method = f"official CAMUS split ({args.official_split_dir})"
    else:
        splits = stratified(meta, args.val_frac, args.test_frac, args.seed)
        method = (f"patient-level, stratified by EF category x worst view quality; "
                  f"val={args.val_frac}, test={args.test_frac}, seed={args.seed}")

    # sanity: disjoint partitions, every listed patient exists on disk
    seen = [p for v in splits.values() for p in v]
    assert len(seen) == len(set(seen)), "patient appears in more than one split"
    unknown = set(seen) - {m["patient"] for m in meta}
    assert not unknown, f"split lists patients not in data root: {sorted(unknown)[:5]}"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps({**splits, "method": method}, indent=2))

    df = pd.DataFrame(meta)
    split_of = {p: s for s, ps in splits.items() for p in ps}
    df["split"] = df["patient"].map(split_of)
    df.to_csv(out / "patients.csv", index=False)

    print(method)
    print(pd.crosstab(df["ef_category"], df["split"], margins=True))


if __name__ == "__main__":
    main()
