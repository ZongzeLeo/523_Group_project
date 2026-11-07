"""Predict masks and save them at NATIVE resolution with the original NIfTI header,
so downstream volume code can read pixel spacing straight from the file.

For each frame, writes to <out-dir>/<patient>/:
    <pid>_<view>_<phase>_pred_raw.nii.gz   argmax of upsampled softmax
    <pid>_<view>_<phase>_pred.nii.gz       + LV post-processing (largest connected
                                           component, holes filled)
Labels match CAMUS GT: 0 background, 1 LV endocardium, 2 myocardium, 3 LA.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from common import LV_LABEL, N_CLASSES, PHASES, VIEWS, image_path, load_nifti
from dataset import preprocess_image, unletterbox_probs
from unet import UNet


def postprocess_lv(mask):
    """Keep the largest LV component and fill holes inside it.
    Returns (mask, n_lv_components_before, n_pixels_changed)."""
    out = mask.copy()
    lv = mask == LV_LABEL
    lab, n = ndimage.label(lv)
    if n > 1:
        sizes = ndimage.sum(lv, lab, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        out[lv & (lab != keep)] = 0
    filled = ndimage.binary_fill_holes(out == LV_LABEL)
    out[filled] = LV_LABEL
    return out, n, int((out != mask).sum())


def save_like(arr2d, ref_img, path):
    data = arr2d.astype(np.uint8).reshape(ref_img.shape)  # restore singleton dims
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(data, ref_img.affine, hdr), str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", default="splits/splits.json")
    ap.add_argument("--checkpoint", default="runs/unet_baseline/best.pt")
    ap.add_argument("--out-dir", default="runs/unet_baseline/predictions")
    ap.add_argument("--split", default="test", help="comma-separated, e.g. 'val,test'")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    size = ckpt["size"]
    model = UNet(n_classes=N_CLASSES, base=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    splits = json.loads(Path(args.splits).read_text())
    out = Path(args.out_dir)
    rows = []
    for split in args.split.split(","):
        for pid in splits[split]:
            (out / pid).mkdir(parents=True, exist_ok=True)
            for view in VIEWS:
                for phase in PHASES:
                    arr, ref, spacing = load_nifti(image_path(args.data_root, pid, view, phase))
                    x, lb = preprocess_image(arr, spacing, size)
                    with torch.no_grad():
                        probs = model(x[None].to(device)).float().softmax(1).cpu()
                        # crop the letterbox and resample probabilities (not labels)
                        # back to the native grid
                        probs = unletterbox_probs(probs, lb)
                    raw = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
                    post, n_comp, n_changed = postprocess_lv(raw)
                    stem = f"{pid}_{view}_{phase}"
                    save_like(raw, ref, out / pid / f"{stem}_pred_raw.nii.gz")
                    save_like(post, ref, out / pid / f"{stem}_pred.nii.gz")
                    rows.append({"patient": pid, "split": split, "view": view, "phase": phase,
                                 "spacing_0_mm": spacing[0], "spacing_1_mm": spacing[1],
                                 "letterbox_iso_mm": lb["iso_spacing_mm"], "letterbox_hw": f"{lb['h']}x{lb['w']}",
                                 "lv_components_raw": n_comp, "postproc_pixels_changed": n_changed})
    pd.DataFrame(rows).to_csv(out / "predictions_manifest.csv", index=False)
    print(f"wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
