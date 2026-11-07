"""Segmentation evaluation at native resolution against CAMUS GT.

Per frame and structure: Dice, Hausdorff distance (HD and HD95, mm) and mean absolute
surface distance (MASD, mm), using pixel spacing from the NIfTI header.
Dice is insensitive to small boundary errors at the apex/annulus, so HD/MASD are
reported alongside it.

Outputs in --out-dir:
    metrics_per_frame.csv   long format: one row per frame x structure x variant
    summary_lv.csv / .md    LV (post-processed) stratified by view/phase, EF category,
                            image quality; plus raw-vs-post-processed comparison
    lv_dice_hd_boxplots.png figure for the report
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from common import LABELS, PHASES, VIEWS, image_path, load_nifti, patient_meta


def dice(p, t):
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else 2.0 * (p & t).sum() / denom


def boundary(m):
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)


def surface_distances(p, t, spacing):
    """HD, HD95 and MASD in mm; NaN if either mask is empty."""
    if not p.any() or not t.any():
        return np.nan, np.nan, np.nan
    bp, bt = boundary(p), boundary(t)
    d_pt = ndimage.distance_transform_edt(~bt, sampling=spacing)[bp]
    d_tp = ndimage.distance_transform_edt(~bp, sampling=spacing)[bt]
    both = np.concatenate([d_pt, d_tp])
    return (float(max(d_pt.max(), d_tp.max())), float(np.percentile(both, 95)),
            float((d_pt.mean() + d_tp.mean()) / 2))


def summarize(df, by):
    g = df.groupby(by, observed=True)
    s = g.agg(n=("dice", "size"),
              dice_mean=("dice", "mean"), dice_sd=("dice", "std"), dice_median=("dice", "median"),
              hd_mean_mm=("hd_mm", "mean"), hd_sd_mm=("hd_mm", "std"),
              hd95_mean_mm=("hd95_mm", "mean"), hd95_sd_mm=("hd95_mm", "std"),
              masd_mean_mm=("masd_mm", "mean"), masd_sd_mm=("masd_mm", "std")).reset_index()
    return s


EF_ORDER = ["<=45", "45-55", ">=55"]
Q_ORDER = ["Good", "Medium", "Poor"]
BLUE, INK, MUTED, GRID = "#2a78d6", "#1a1a19", "#6b6a63", "#e6e5df"


def boxpanel(ax, groups, labels, ylabel):
    data = [g[~np.isnan(g)] for g in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, linewidth=2),
                    boxprops=dict(facecolor=BLUE, edgecolor=BLUE, alpha=0.35),
                    whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=BLUE,
                                    markeredgecolor="none", alpha=0.6))
    ax.set_xticks(range(1, len(labels) + 1),
                  [f"{l}\n(n={len(d)})" for l, d in zip(labels, data)])
    ax.set_ylabel(ylabel, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    return bp


def plot(lv, path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey="row")
    vp = [(v, p) for v in VIEWS for p in PHASES]
    strat = [
        ("View / phase", [f"{v} {p}" for v, p in vp],
         [(lv.view == v) & (lv.phase == p) for v, p in vp]),
        ("LVEF category (%)", [c for c in EF_ORDER if (lv.ef_category == c).any()],
         [lv.ef_category == c for c in EF_ORDER if (lv.ef_category == c).any()]),
        ("Image quality", [q for q in Q_ORDER if (lv.quality == q).any()],
         [lv.quality == q for q in Q_ORDER if (lv.quality == q).any()]),
    ]
    for col, (title, labels, masks) in enumerate(strat):
        boxpanel(axes[0, col], [lv.loc[m, "dice"].to_numpy() for m in masks], labels,
                 "LV Dice" if col == 0 else "")
        boxpanel(axes[1, col], [lv.loc[m, "hd_mm"].to_numpy() for m in masks], labels,
                 "LV Hausdorff distance (mm)" if col == 0 else "")
        axes[0, col].set_title(title, color=INK, fontsize=11, loc="left")
    fig.suptitle("U-Net LV segmentation on CAMUS test set (post-processed masks)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--pred-dir", default="runs/unet_baseline/predictions")
    ap.add_argument("--out-dir", default="runs/unet_baseline/eval")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pids = json.loads(Path(args.splits).read_text())[args.split]

    rows = []
    for pid in pids:
        meta = patient_meta(args.data_root, pid)
        for view in VIEWS:
            for phase in PHASES:
                gt, _, spacing = load_nifti(image_path(args.data_root, pid, view, phase, gt=True))
                px_area = spacing[0] * spacing[1]
                for variant, suffix in (("post", "_pred"), ("raw", "_pred_raw")):
                    pred, _, _ = load_nifti(Path(args.pred_dir) / pid /
                                            f"{pid}_{view}_{phase}{suffix}.nii.gz")
                    assert pred.shape == gt.shape, f"shape mismatch {pid} {view} {phase}"
                    for label, name in LABELS.items():
                        if label == 0:
                            continue
                        p, t = pred == label, gt == label
                        hd, hd95, masd = surface_distances(p, t, spacing)
                        rows.append({
                            "patient": pid, "view": view, "phase": phase, "variant": variant,
                            "structure": name, "ef": meta["ef"],
                            "ef_category": meta["ef_category"],
                            "quality": meta[f"quality_{view}"],
                            "dice": dice(p, t), "hd_mm": hd, "hd95_mm": hd95, "masd_mm": masd,
                            "gt_area_mm2": t.sum() * px_area, "pred_area_mm2": p.sum() * px_area,
                        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_per_frame.csv", index=False)

    lv = df[(df.structure == "lv_endo") & (df.variant == "post")].copy()
    lv["ef_category"] = pd.Categorical(lv.ef_category, EF_ORDER, ordered=True)
    lv["quality"] = pd.Categorical(lv.quality, Q_ORDER, ordered=True)

    tables = {
        "Overall": summarize(lv.assign(all="all"), ["all"]),
        "By view and phase": summarize(lv, ["view", "phase"]),
        "By LVEF category": summarize(lv, ["ef_category"]),
        "By image quality": summarize(lv, ["quality"]),
        "Raw vs post-processed (LV)": summarize(df[df.structure == "lv_endo"], ["variant"]),
        "All structures (post-processed)": summarize(df[df.variant == "post"], ["structure"]),
    }
    pd.concat([t.assign(table=k) for k, t in tables.items()]).to_csv(out / "summary_lv.csv",
                                                                     index=False)
    md = [f"# LV segmentation results ({args.split} split, {len(pids)} patients)\n"]
    for k, t in tables.items():
        md += [f"## {k}\n", t.to_markdown(index=False, floatfmt=".3f"), ""]
    (out / "summary_lv.md").write_text("\n".join(md))
    print("\n".join(md))

    plot(lv, out / "lv_dice_hd_boxplots.png")
    print(f"wrote results to {out}")


if __name__ == "__main__":
    main()
