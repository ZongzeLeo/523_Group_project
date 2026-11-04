"""Reuse original notebook frame profiles across repeated paired-mask scenarios.

The original functions perform every numerical operation. Only their per-frame
outputs are cached. Copies protect diameter arrays from the original volume
function's in-place mm-to-m conversion.
"""
import warnings
import numpy as np

from . import GeometryError
from .publisher import publisher_functions


def publisher_profile(frame):
    mask = np.asarray(frame["mask"])
    spacing = np.asarray(frame["geometry"]["spacing_xy_mm"], dtype=float)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all() or mask.sum() < 10:
        raise GeometryError("Publisher profile requires a nonempty binary LV mask")
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise GeometryError("Invalid native profile spacing")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            diameters, step = publisher_functions()["_compute_diameters"](mask.astype(bool), tuple(spacing[::-1]))
        except (ValueError, IndexError, UnboundLocalError, ZeroDivisionError, OverflowError) as exc:
            raise GeometryError(f"Original CAMUS profile failed: {exc}") from exc
    return {"diameters_mm": diameters, "step_mm": step, "spacing_xy_mm": tuple(spacing),
            "warnings": sorted(set(str(w.message) for w in recorded))}


def measure_profiles(profiles):
    required = {f"{v}_{p}" for v in ("2CH", "4CH") for p in ("ED", "ES")}
    if set(profiles) != required or any(v is None for v in profiles.values()):
        raise GeometryError("All four original publisher profiles are required")
    for view in ("2CH", "4CH"):
        if profiles[f"{view}_ED"]["spacing_xy_mm"] != profiles[f"{view}_ES"]["spacing_xy_mm"]:
            raise GeometryError("Original calculation assumes equal within-view ED/ES spacing")
    step = max(profiles[k]["step_mm"] for k in ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"))
    original = publisher_functions()["_compute_left_ventricle_volume_by_instant"]
    volumes = [original(profiles[f"2CH_{phase}"]["diameters_mm"].copy(),
                        profiles[f"4CH_{phase}"]["diameters_mm"].copy(), step) for phase in ("ED", "ES")]
    edv, esv = volumes
    if not np.isfinite(volumes).all() or edv <= 0 or esv < 0:
        raise GeometryError("Original CAMUS profiles produced invalid volumes")
    continuous = 100*(edv-esv)/edv
    return {"edv_ml": int(edv), "esv_ml": int(esv), "ef_pct": int(round(continuous)),
            "ef_before_final_rounding_pct": float(continuous),
            "status": "invalid_volumes" if esv > edv else "ok",
            "warnings": sorted({w for p in profiles.values() for w in p["warnings"]})}
