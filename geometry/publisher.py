"""Execute the reviewed, hash-pinned original CAMUS notebook calculation.

Only its import and function-definition cells are loaded. No sample-patient or
interactive cells run. The original numeric function bodies are unchanged.
"""

from functools import lru_cache
import json
from pathlib import Path
import warnings

import numpy as np

from data.download_camus import sha256
from . import GeometryError

NOTEBOOK = Path(__file__).resolve().parent / "vendor/script_camus_ef.ipynb"
NOTEBOOK_SHA256 = "09cef7321d43a4986089563b088af1ffd18a93de5c545e84a27f01311c3a4c0c"
FUNCTION_CELLS = (2, 4, 5, 6, 8)


@lru_cache(maxsize=1)
def publisher_functions():
    if sha256(NOTEBOOK) != NOTEBOOK_SHA256:
        raise ValueError("CAMUS notebook differs from the reviewed original; refusing execution")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {}
    for index in FUNCTION_CELLS:
        cell = cells[index]
        if cell["cell_type"] != "code":
            raise ValueError("Publisher notebook cell layout changed")
        exec(compile("".join(cell["source"]), f"CAMUS_original_cell_{index}", "exec"), namespace)
    return namespace


def measure_publisher(frames):
    """Four native binary masks and spacing -> original rounded volumes and EF.

    Accepts the data loader's native-frame dictionaries. No reference label is
    accepted. Missing/degenerate masks fail explicitly; no cleanup occurs here.
    """
    keys = {f"{view}_{phase}" for view in ("2CH", "4CH") for phase in ("ED", "ES")}
    if set(frames) != keys:
        raise GeometryError("Publisher calculation requires all four paired frames")
    masks, spacings = {}, {}
    for key, frame in frames.items():
        mask = np.asarray(frame["mask"])
        spacing = np.asarray(frame["geometry"]["spacing_xy_mm"], dtype=float)
        if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or mask.sum() < 10:
            raise GeometryError(f"{key}: empty, tiny, or nonbinary LV mask")
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise GeometryError(f"{key}: invalid native spacing")
        masks[key], spacings[key] = mask.astype(bool), tuple(spacing[::-1])
    for view in ("2CH", "4CH"):
        if spacings[f"{view}_ED"] != spacings[f"{view}_ES"]:
            raise GeometryError("Original calculation assumes identical ED/ES spacing within each view")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            edv, esv = publisher_functions()["compute_left_ventricle_volumes"](
                masks["2CH_ED"], masks["2CH_ES"], spacings["2CH_ED"],
                masks["4CH_ED"], masks["4CH_ES"], spacings["4CH_ED"])
        except (ValueError, IndexError, UnboundLocalError, ZeroDivisionError, OverflowError) as exc:
            raise GeometryError(f"Original CAMUS calculation failed: {exc}") from exc
    if not np.isfinite([edv, esv]).all() or edv <= 0 or esv < 0:
        raise GeometryError("Original CAMUS calculation produced invalid volumes")
    continuous = 100 * (edv - esv) / edv
    return {"edv_ml": int(edv), "esv_ml": int(esv), "ef_pct": int(round(continuous)),
            "ef_before_final_rounding_pct": float(continuous),
            "status": "invalid_volumes" if esv > edv else "ok",
            "warnings": sorted(set(str(item.message) for item in recorded))}
