"""Shared CAMUS helpers: file layout, metadata parsing, label definitions.

Expected CAMUS layout (public NIfTI release):
    <data_root>/patient0001/patient0001_2CH_ED.nii.gz
    <data_root>/patient0001/patient0001_2CH_ED_gt.nii.gz
    <data_root>/patient0001/Info_2CH.cfg
    ...
"""
from pathlib import Path

import nibabel as nib
import numpy as np

VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")
# CAMUS ground-truth labels
LABELS = {0: "background", 1: "lv_endo", 2: "lv_myo", 3: "la"}
LV_LABEL = 1
N_CLASSES = 4
QUALITY_ORDER = {"Good": 0, "Medium": 1, "Poor": 2}


def ef_category(ef: float) -> str:
    """CAMUS LVEF categories used for stratification."""
    if ef <= 45:
        return "<=45"
    if ef < 55:
        return "45-55"
    return ">=55"

#returns a list of patient IDs
def patient_ids(data_root) -> list[str]:
    return sorted(p.name for p in Path(data_root).iterdir()
                  if p.is_dir() and p.name.startswith("patient"))

#builds the file path for one frame
def image_path(data_root, pid, view, phase, gt=False) -> Path:
    suffix = "_gt" if gt else ""
    return Path(data_root) / pid / f"{pid}_{view}_{phase}{suffix}.nii.gz"


def read_info(path) -> dict:
    """Parse a CAMUS Info_*.cfg file ('Key: value' per line)."""
    info = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = (s.strip() for s in line.split(":", 1))
        try:
            info[key] = float(val)
        except ValueError:
            info[key] = val
    return info


def patient_meta(data_root, pid) -> dict:
    """Per-patient metadata: reference EF/volumes and per-view image quality."""
    infos = {v: read_info(Path(data_root) / pid / f"Info_{v}.cfg") for v in VIEWS}
    i2 = infos["2CH"]
    ef = i2.get("LVef", i2.get("EF"))
    meta = {
        "patient": pid,
        "ef": ef,
        "ef_category": ef_category(ef) if ef is not None else None,
        "edv_ref": i2.get("LVedv"),
        "esv_ref": i2.get("LVesv"),
    }
    for v in VIEWS:
        meta[f"quality_{v}"] = infos[v].get("ImageQuality")
    return meta


def load_nifti(path):
    """Return (2D array, nibabel image, (spacing_axis0, spacing_axis1) in mm)."""
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj).squeeze()
    spacing = tuple(float(z) for z in img.header.get_zooms()[:2])
    return arr, img, spacing
