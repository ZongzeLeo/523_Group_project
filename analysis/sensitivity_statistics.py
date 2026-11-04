"""Paired patient summaries and descriptive, patient-level associations."""
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from .perturbations import METHODS, FRAME_KEYS
from .statistics import bootstrap_mean_ci, geometry_metrics

QUANTITIES = ("edv_ml", "esv_ml", "ef_pp")
FEATURES = (
    ("area_error_pct", "ef_error_vs_gt_pp"),
    ("length_error_mm", "ef_error_vs_gt_pp"),
    ("abs_area_error_pct", "absolute_ef_error_pp"),
    ("abs_length_error_mm", "absolute_ef_error_pp"),
    ("apex_displacement_mm", "absolute_ef_error_pp"),
    ("base_midpoint_displacement_mm", "absolute_ef_error_pp"),
    ("one_minus_dice", "absolute_ef_error_pp"),
)


def groups(rows):
    yield "overall", "all", rows
    for field in ("reference_ef_group", "image_quality"):
        for value in sorted({r[field] for r in rows}):
            yield field, value, [r for r in rows if r[field] == value]


def paired_summary(rows, config):
    selected = defaultdict(list)
    for row in rows:
        selected[(row["method"], row["scenario_id"])].append(row)
    output = []
    for (method, scenario), records in sorted(selected.items()):
        for group_type, group, subset in groups(records):
            if len({r["patient_id"] for r in subset}) != len(subset):
                raise ValueError("Repeated patient within scenario summary")
            first = subset[0]
            row = {"method": method, "scenario_id": scenario,
                   **{k: first[k] for k in ("family", "magnitude_mm", "view_scope", "phase_scope")},
                   "group_type": group_type, "group": group, "attempted_patients": len(subset),
                   "failed_geometry_patients": sum(r["status"] == "invalid_geometry" for r in subset),
                   "invalid_volume_patients": sum(r["status"] == "invalid_volumes" for r in subset),
                   "border_contact_patients": sum(r["border_contact"] for r in subset)}
            for quantity in QUANTITIES:
                values = [r["delta_"+quantity] for r in subset if r["delta_"+quantity] is not None]
                row[quantity+"_numeric_pairs"] = len(values)
                for label, x in (("mean", values), ("mae", np.abs(values))):
                    row[quantity+"_"+label] = float(np.mean(x)) if len(x) else None
                    lo, hi = bootstrap_mean_ci(x, seed=config["seed"], repetitions=config["bootstrap_repetitions"])
                    row[quantity+"_"+label+"_ci95_low"] = lo
                    row[quantity+"_"+label+"_ci95_high"] = hi
            output.append(row)
    return output


def method_contrasts(rows, config):
    index = {(r["patient_id"], r["scenario_id"], r["method"]): r for r in rows}
    patients = sorted({r["patient_id"] for r in rows})
    scenarios = sorted({r["scenario_id"] for r in rows})
    output = []
    for scenario in scenarios:
        for first, second in combinations(METHODS, 2):
            for quantity in QUANTITIES:
                key = "delta_"+quantity
                paired = [(index[p, scenario, first][key], index[p, scenario, second][key]) for p in patients]
                values = [a-b for a, b in paired if a is not None and b is not None]
                low, high = bootstrap_mean_ci(values, seed=config["seed"], repetitions=config["bootstrap_repetitions"])
                output.append({"scenario_id": scenario, "method_a": first, "method_b": second,
                               "quantity": quantity, "attempted_patients": len(patients), "numeric_pairs": len(values),
                               "mean_delta_a_minus_b": float(np.mean(values)) if values else None,
                               "ci95_low": low, "ci95_high": high})
    return output


def spearman_ci(x, y, seed=523, repetitions=2000):
    """Re-rank each patient bootstrap sample, including its duplicate ranks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite([x, y]).all():
        raise ValueError("Association requires paired finite observations")
    result = {"n": len(x), "spearman_rho": None, "ci95_low": None, "ci95_high": None,
              "valid_bootstrap_replicates": 0}
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    result["spearman_rho"] = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    index = np.random.default_rng(seed).integers(0, len(x), size=(repetitions, len(x)))
    a, b = rankdata(x[index], axis=1), rankdata(y[index], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    valid = denominator > 0
    rho = (a[valid]*b[valid]).sum(axis=1)/denominator[valid]
    result["valid_bootstrap_replicates"] = int(valid.sum())
    if len(rho):
        result["ci95_low"], result["ci95_high"] = np.percentile(rho, [2.5, 97.5]).tolist()
    return result


def observed_summaries(frames, patients, config):
    indexed = {(r["patient_id"], r["method"]): r for r in patients}
    associations, comparisons = [], []
    for method in METHODS:
        selected = [r for r in patients if r["method"] == method]
        for group_type, group, subset in groups(selected):
            comparisons.append({"method": method, "group_type": group_type, "group": group,
                                **geometry_metrics(subset)})
        for view_phase in FRAME_KEYS:
            records = []
            for frame in frames:
                if frame["view_phase"] != view_phase:
                    continue
                patient = indexed[frame["patient_id"], method]
                # Use patient-level worst-view quality for all stratified associations.
                records.append({**frame, "image_quality": patient["image_quality"],
                                "ef_error_vs_gt_pp": patient["ef_error_vs_gt_pp"],
                                "absolute_ef_error_pp": abs(patient["ef_error_vs_gt_pp"])
                                if patient["ef_error_vs_gt_pp"] is not None else None})
            for group_type, group, subset in groups(records):
                for feature, target in FEATURES:
                    pairs = [r for r in subset if r[feature] is not None and r[target] is not None]
                    if len({r["patient_id"] for r in pairs}) != len(pairs):
                        raise ValueError("Association contains repeated patients")
                    stats = spearman_ci([r[feature] for r in pairs], [r[target] for r in pairs],
                                        config["seed"], config["bootstrap_repetitions"])
                    associations.append({"method": method, "view_phase": view_phase, "feature": feature,
                                         "target": target, "group_type": group_type, "group": group,
                                         "attempted_patients": len(subset), **stats})
    return comparisons, associations


def volume_error_contributions(gt_edv, gt_esv, predicted_edv, predicted_esv, ef_error_pp):
    """Exact arithmetic decomposition; the remainder isolates final EF rounding."""
    if min(gt_edv, predicted_edv) <= 0:
        raise ValueError("EDV must be positive")
    edv = 100*gt_esv*(predicted_edv-gt_edv)/(gt_edv*predicted_edv)
    esv = -100*(predicted_esv-gt_esv)/predicted_edv
    return {"edv_contribution_pp": edv, "esv_contribution_pp": esv,
            "ef_rounding_remainder_pp": ef_error_pp-edv-esv}
