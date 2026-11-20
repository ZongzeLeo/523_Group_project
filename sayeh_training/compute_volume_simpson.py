"""LV volumes and EF from Sayeh's predicted masks, using the team repo's geometry code.

By default runs only Simpson's biplane (--methods simpson). Available methods,
each run on each test patient, on both the predicted masks and the
CAMUS expert masks, so segmentation-induced errors can be read off directly:
  publisher   : original CAMUS notebook calculation (geometry/publisher.py)
  simpson     : phase-specific Simpson's biplane (geometry/comparators.py)
  area_length : biplane area-length (geometry/comparators.py)

Run from the REPO ROOT (so `geometry` and `data` import):
    python sayeh_training/compute_volumes.py \
        --data-root /data/scratch/users/sayeh/CAMUS/database_nifti \
        --pred-dir sayeh_training/runs/unet_e80/predictions \
        --splits sayeh_training/splits/splits.json \
        --out sayeh_training/runs/unet_e80/eval/volumes_per_case.csv \
        --case-table sayeh_training/runs/unet_e80/eval/per_case_results.csv

Environment: simpson and area_length run in any env with scikit-image. The
`publisher` method (only if requested) executes the original CAMUS notebook, which
needs the repo's pinned versions (numpy<2.1, SimpleITK; see requirements.txt).
With newer numpy it fails and its columns read "failed: ..."; simpson and
area_length work either way. A separate env avoids touching test_env:
    conda create -n camus_geom python=3.12 -y && conda activate camus_geom
    pip install -r requirements.txt pandas

Masks are loaded with nibabel (x, y axes) and transposed to (y, x) with spacing
(x, y) in mm, matching the repo's data/native.py convention.
"""
import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from geometry import GeometryError                                    # noqa: E402
from geometry.comparators import measure_mask, measure_patient        # noqa: E402

ALL_METHODS = ("simpson", "area_length", "publisher")

FRAMES = [f"{v}_{p}" for v in ("2CH", "4CH") for p in ("ED", "ES")]


def load_lv(path):
    """nibabel (x, y[,1]) label map -> (y, x) binary LV mask, spacing (x, y) mm."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj).squeeze()
    if img.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError(f"{path.name}: spatial units are not mm")
    spacing_xy = [float(z) for z in img.header.get_zooms()[:2]]
    return (arr == 1).T.astype(np.uint8), spacing_xy


def patient_frames(paths):
    return {k: {"mask": m, "geometry": {"spacing_xy_mm": s}}
            for k, (m, s) in ((k, load_lv(p)) for k, p in paths.items())}


def measure_all(frames, methods):
    """Requested methods on one set of four frames; failures recorded, not raised."""
    out = {}
    if "publisher" in methods:
        from geometry.publisher import measure_publisher
        try:
            r = measure_publisher(frames)
            out.update(publisher_edv_ml=r["edv_ml"], publisher_esv_ml=r["esv_ml"],
                       publisher_ef_pct=r["ef_before_final_rounding_pct"], publisher_status=r["status"])
        except GeometryError as e:
            out.update(publisher_status=f"failed: {e}")
    comp = [m for m in methods if m in ("simpson", "area_length")]
    if comp:
        try:
            measured = {k: measure_mask(f["mask"], f["geometry"]["spacing_xy_mm"]) for k, f in frames.items()}
            res = measure_patient(measured)
            names = {"simpson": "simpson_biplane", "area_length": "area_length_biplane_ellipsoid"}
            for key in comp:
                r = res[names[key]]
                out.update({f"{key}_edv_ml": r["edv_ml"], f"{key}_esv_ml": r["esv_ml"],
                            f"{key}_ef_pct": r["ef_pct"], f"{key}_status": r["status"]})
            out["geometry_warnings"] = ";".join(sorted({w for k in comp for w in res[names[k]]["warnings"]}))
        except GeometryError as e:
            out.update({f"{k}_status": f"failed: {e}" for k in comp})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred-dir", default="sayeh_training/runs/unet_e80/predictions")
    ap.add_argument("--splits", default="sayeh_training/splits/splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default="simpson",
                    help="comma-separated from simpson, area_length, publisher (default: simpson)")
    ap.add_argument("--suffix", default="_pred", help="'_pred' (post-processed) or '_pred_raw'")
    ap.add_argument("--out", default="sayeh_training/runs/unet_e80/eval/volumes_per_case.csv")
    ap.add_argument("--case-table", default=None,
                    help="optional per_case_results.csv to merge with (writes *_with_volumes.csv)")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    bad = set(methods) - set(ALL_METHODS)
    if bad:
        ap.error(f"unknown methods {bad}; choose from {ALL_METHODS}")
    if "publisher" in methods and tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 1):
        print(f"note: numpy {np.__version__} >= 2.1, so the publisher method will fail (needs numpy<2.1)")
    pids = json.loads(Path(args.splits).read_text())[args.split]
    rows = []
    for i, pid in enumerate(pids, 1):
        row = {"patient": pid}
        sources = {
            "pred": {k: Path(args.pred_dir) / pid / f"{pid}_{k}{args.suffix}.nii.gz" for k in FRAMES},
            "gt": {k: Path(args.data_root) / pid / f"{pid}_{k}_gt.nii.gz" for k in FRAMES},
        }
        for src, paths in sources.items():
            row.update({f"{src}_{k}": v for k, v in measure_all(patient_frames(paths), methods).items()})
        rows.append(row)
        if i % 10 == 0 or i == len(pids):
            print(f"{i}/{len(pids)} patients", flush=True)

    df = pd.DataFrame(rows)
    # segmentation-induced error: predicted minus expert-mask value, same method
    for method in methods:
        for q in ("edv_ml", "esv_ml", "ef_pct"):
            p, g = f"pred_{method}_{q}", f"gt_{method}_{q}"
            if p in df and g in df:
                df[f"err_{method}_{q}"] = df[p] - df[g]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.3f")
    print(f"wrote {args.out}")

    summary = []
    for method in methods:
        for q in ("edv_ml", "esv_ml", "ef_pct"):
            c = f"err_{method}_{q}"
            if c in df and df[c].notna().any():
                summary.append({"method": method, "quantity": q, "n": int(df[c].notna().sum()),
                                "bias": df[c].mean(), "MAE": df[c].abs().mean()})
    print(pd.DataFrame(summary).to_string(index=False, float_format="%.2f"))
    failed = df.filter(like="_status").apply(lambda s: s.astype(str).str.startswith("failed")).sum()
    if failed.any():
        print("failures per method:\n" + failed[failed > 0].to_string())

    if args.case_table:
        merged = pd.read_csv(args.case_table).merge(df, on="patient", how="left")
        out2 = Path(args.case_table).with_name(Path(args.case_table).stem + "_with_volumes.csv")
        merged.to_csv(out2, index=False, float_format="%.4f")
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()