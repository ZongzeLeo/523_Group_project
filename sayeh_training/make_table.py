"""Per-case (one row per patient) results table.

Combines CAMUS patient info (sex, age, image quality, reference EF, frame info)
with LV Dice for each of the four frames (2CH/4CH x ED/ES),
read from evaluate_dice.py's metrics_per_frame.csv.

Usage:
    python make_case_table.py --data-root $D --metrics runs/unet_e80/eval/metrics_per_frame.csv \
        --out runs/unet_e80/eval/per_case_results.csv

Note: CAMUS does not provide reference EDV/ESV. Volumes come from the geometric
(Simpson / area-length) step and can be merged on the `patient` column.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import PHASES, VIEWS, ef_category, read_info

METRICS = ["dice"]  # LV Dice for each of the four frames


def patient_info(data_root, pid):
    infos = {v: read_info(Path(data_root) / pid / f"Info_{v}.cfg") for v in VIEWS}
    i2 = infos["2CH"]
    ef = i2.get("EF", i2.get("LVef"))
    row = {"patient": pid, "sex": i2.get("Sex"), "age": i2.get("Age"),
           "ef_ref": ef, "ef_category": ef_category(ef) if ef is not None else None}
    for v in VIEWS:
        row[f"quality_{v}"] = infos[v].get("ImageQuality")
        row[f"frame_rate_{v}"] = infos[v].get("FrameRate")
        row[f"n_frames_{v}"] = infos[v].get("NbFrame")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--metrics", default="runs/unet_e80/eval/metrics_per_frame.csv")
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--out", default="runs/unet_e80/eval/per_case_results.csv")
    ap.add_argument("--variant", default="post", choices=["post", "raw"])
    args = ap.parse_args()

    m = pd.read_csv(args.metrics)
    m = m[(m.structure == "lv_endo") & (m.variant == args.variant)].copy()
    m["frame"] = m.view + "_" + m.phase

    # wide: one row per patient, one column per metric x frame (e.g. dice_2CH_ED)
    wide = m.pivot(index="patient", columns="frame", values=[c for c in METRICS if c in m])
    wide.columns = [f"{metric}_{frame}" for metric, frame in wide.columns]
    order = [f"{metric}_{v}_{p}" for metric in METRICS for v in VIEWS for p in PHASES]
    wide = wide[[c for c in order if c in wide.columns]].reset_index()

    splits = json.loads(Path(args.splits).read_text())
    split_of = {p: s for s in ("train", "val", "test") for p in splits.get(s, [])}
    info = pd.DataFrame([patient_info(args.data_root, p) for p in wide.patient])
    info.insert(1, "split", info.patient.map(split_of))

    table = info.merge(wide, on="patient")
    for c in ["age"] + [f"n_frames_{v}" for v in VIEWS] + [f"frame_rate_{v}" for v in VIEWS]:
        table[c] = pd.to_numeric(table[c], errors="coerce").round().astype("Int64")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, float_format="%.4f")
    print(f"wrote {len(table)} patients x {table.shape[1]} columns to {args.out}")
    print(table.iloc[:3, :12].to_string(index=False))


if __name__ == "__main__":
    main()