"""Controlled native-grid mask perturbations in physical millimetres.

The coordinate system is fixed by the unperturbed expert mask. Geometry is
redetected after perturbation. No smoothing, component selection or hole filling
is performed. See docs/perturbation_protocol.md for the complete experiment.
"""
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
from scipy.ndimage import map_coordinates

from geometry import GeometryError

METHODS = ("publisher_original", "simpson_biplane", "area_length_biplane_ellipsoid")
FRAME_KEYS = ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES")


@dataclass(frozen=True)
class Scenario:
    family: str
    magnitude_mm: float
    view_scope: str
    phase_scope: str

    @property
    def scenario_id(self):
        if self.family == "control":
            return "control"
        value = ("p" if self.magnitude_mm > 0 else "n") + f"{abs(self.magnitude_mm):g}"
        return f"{self.family}_{value}mm_{self.view_scope}_{self.phase_scope}"

    def affects(self, key):
        if key not in FRAME_KEYS:
            raise ValueError(f"Unknown paired frame: {key}")
        view, phase = key.split("_")
        return (self.family != "control" and self.view_scope in (view, "both")
                and self.phase_scope in (phase, "both"))

    def record(self):
        return {"scenario_id": self.scenario_id, **asdict(self)}


def scenarios(config):
    """One shared control plus the declared view/phase factorial design."""
    if config["protocol_version"] != 1 or config["split"] != "test":
        raise ValueError("Protocol v1 evaluates the frozen test split")
    if tuple(config["methods"]) != METHODS:
        raise ValueError("Protocol v1 requires the three specified methods")
    if config["view_scopes"] != ["2CH", "4CH", "both"] or config["phase_scopes"] != ["ED", "ES", "both"]:
        raise ValueError("Incomplete view/phase experiment")
    apex = config["apical_truncation_mm"]
    base = config["basal_displacement_mm"]
    taper = config["basal_taper_mm"]
    if (not np.isfinite([*apex, *base, taper]).all() or taper <= 0
            or any(v <= 0 for v in apex) or any(v == 0 or abs(v) >= taper for v in base)):
        raise ValueError("Invalid physical perturbation magnitude or taper")
    if config["bootstrap_repetitions"] < 100 or config["prediction_threshold"] != 0.5:
        raise ValueError("Invalid uncertainty or fixed prediction threshold")
    result = [Scenario("control", 0.0, "none", "none")]
    for family, values in (("apical_truncation", apex), ("basal_displacement", base)):
        for value, view, phase in product(values, config["view_scopes"], config["phase_scopes"]):
            result.append(Scenario(family, float(value), view, phase))
    if len({s.scenario_id for s in result}) != len(result):
        raise ValueError("Duplicate perturbation scenarios")
    return result


def axial_coordinates(shape, spacing_xy_mm, landmarks):
    spacing = np.asarray(spacing_xy_mm, dtype=float)
    base = np.asarray(landmarks["base_midpoint_xy_mm"], dtype=float)
    apex = np.asarray(landmarks["apex_xy_mm"], dtype=float)
    if (spacing.shape != (2,) or base.shape != (2,) or apex.shape != (2,)
            or not np.isfinite([*spacing, *base, *apex]).all() or (spacing <= 0).any()):
        raise GeometryError("Invalid perturbation coordinate system")
    length = float(np.linalg.norm(apex-base))
    if length <= 0:
        raise GeometryError("Degenerate perturbation axis")
    axis = (apex-base)/length
    yy, xx = np.indices(shape, dtype=float)
    axial = (xx*spacing[0]-base[0])*axis[0] + (yy*spacing[1]-base[1])*axis[1]
    return axial, axis, length


def perturb_mask(mask, spacing_xy_mm, landmarks, family, magnitude_mm, basal_taper_mm=10.0):
    """Return a new binary mask; positive basal displacement is toward the apex.

    Apical truncation retains z <= L-d. Basal displacement uses z'=z+d*w(z),
    w=1 below the base, linearly decreasing to zero at the taper distance.
    Its analytic inverse is sampled with nearest-neighbour interpolation.
    """
    original = np.asarray(mask)
    if original.ndim != 2 or not np.isin(original, [0, 1]).all() or original.sum() < 10:
        raise GeometryError("Perturbation requires a nonempty binary 2D mask")
    if not np.isfinite([magnitude_mm, basal_taper_mm]).all():
        raise GeometryError("Perturbation magnitudes must be finite")
    if family not in ("control", "apical_truncation", "basal_displacement"):
        raise GeometryError(f"Unknown perturbation family: {family}")
    if family == "control" and magnitude_mm != 0:
        raise GeometryError("Control displacement must be zero")
    mask = original.astype(bool)
    z, axis, length = axial_coordinates(mask.shape, spacing_xy_mm, landmarks)
    if magnitude_mm == 0:
        changed = mask.copy()
    elif family == "apical_truncation":
        if not 0 < magnitude_mm < length:
            raise GeometryError("Apical truncation must lie between zero and the long-axis length")
        changed = mask & (z <= length-magnitude_mm)
    else:
        if not 0 < abs(magnitude_mm) < basal_taper_mm < length:
            raise GeometryError("Basal deformation requires |d| < taper < long-axis length")
        d, band = magnitude_mm, basal_taper_mm
        source_z = np.where(z < d, z-d, np.where(z < band, (z-d)/(1-d/band), z))
        shift = source_z-z
        yy, xx = np.indices(mask.shape, dtype=float)
        coords = [yy+shift*axis[1]/spacing_xy_mm[1], xx+shift*axis[0]/spacing_xy_mm[0]]
        changed = map_coordinates(mask.astype(np.uint8), coords, order=0, mode="constant", cval=0,
                                  prefilter=False).astype(bool)
        # Prevent floating-point indexing from changing the undeformed region.
        changed[z >= band] = mask[z >= band]
    changed_z, original_z = z[changed], z[mask]
    audit = {
        "requested_magnitude_mm": float(magnitude_mm),
        "changed_pixels": int(np.count_nonzero(changed != mask)),
        "area_change_mm2": float((int(changed.sum())-int(mask.sum()))*np.prod(spacing_xy_mm)),
        "apical_extent_change_mm": float(changed_z.max()-original_z.max()) if changed_z.size else None,
        "basal_extent_change_mm": float(changed_z.min()-original_z.min()) if changed_z.size else None,
        "touches_border": bool(changed[[0, -1]].any() or changed[:, [0, -1]].any()),
        "empty": not bool(changed.any()),
    }
    return changed.astype(np.uint8), audit
