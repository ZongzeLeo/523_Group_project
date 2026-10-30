"""Descriptive agreement and patient-bootstrap uncertainty with coverage counts."""

from collections import Counter
import numpy as np

METHODS = ("publisher_original",)


def describe(values):
    values = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(values):
        return {"n": 0, "mean": None, "sd": None, "median": None, "p05": None, "p95": None}
    return {"n": len(values), "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values)>1 else None,
            "median": float(np.median(values)), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def bootstrap_mean_ci(values, seed=523, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    if not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite patient-level values")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def agreement(values):
    values = np.asarray(values, dtype=float)
    result = {"n": len(values), "mae": None, "rmse": None, "bias": None, "loa_lower": None, "loa_upper": None}
    if not len(values):
        return result
    result.update(mae=float(np.abs(values).mean()), rmse=float(np.sqrt((values**2).mean())), bias=float(values.mean()))
    if len(values)>1:
        sd = float(values.std(ddof=1))
        result.update(loa_lower=result["bias"]-1.96*sd, loa_upper=result["bias"]+1.96*sd)
    return result


def geometry_metrics(rows):
    pairs = [row for row in rows if row["predicted_ef_pct"] is not None]
    result = {"attempted_patients": len(rows), "numeric_pairs": len(pairs),
              "failed_geometry_patients": sum(row["status"] == "invalid_geometry" for row in rows),
              "invalid_volume_patients": sum(row["status"] == "invalid_volumes" for row in rows),
              "warning_patients": sum(bool(row["warnings"]) for row in rows)}
    for field, unit in (("ef_error_vs_camus_pp", "ef_vs_camus_pp"), ("ef_error_vs_gt_pp", "ef_vs_gt_pp"),
                        ("edv_error_vs_gt_ml", "edv_vs_gt_ml"), ("esv_error_vs_gt_ml", "esv_vs_gt_ml")):
        stats = agreement([row[field] for row in pairs])
        result.update({f"{unit}_{name}": value for name, value in stats.items() if name != "n"})
    errors = [abs(row["ef_error_vs_camus_pp"]) for row in pairs]
    result["ef_mae_vs_camus_ci95_low"], result["ef_mae_vs_camus_ci95_high"] = bootstrap_mean_ci(errors)
    result["pearson_r_vs_camus"] = None
    if len(pairs)>1:
        p = [row["predicted_ef_pct"] for row in pairs]
        g = [row["camus_reference_ef_pct"] for row in pairs]
        if np.std(p)>0 and np.std(g)>0:
            result["pearson_r_vs_camus"] = float(np.corrcoef(p, g)[0, 1])
    return result


def stratify_geometry(rows):
    metrics = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        groups = [("overall", "all", selected)]
        for field in ("reference_ef_group", "image_quality"):
            for value in sorted({row[field] for row in selected}):
                groups.append((field, value, [row for row in selected if row[field] == value]))
        for group_type, group, subset in groups:
            metrics.append({"method": method, "group_type": group_type, "group": group, **geometry_metrics(subset)})
    return metrics


def stratify_frames(rows):
    groups = [("overall", "all", rows)]
    for field in ("view", "phase", "view_phase", "image_quality", "reference_ef_group"):
        for value in sorted({row[field] for row in rows}):
            groups.append((field, value, [row for row in rows if row[field] == value]))
    result = []
    for group_type, group, subset in groups:
        record = {"group_type": group_type, "group": group, "frames": len(subset),
                  "empty_predictions": sum(row["empty_prediction"] for row in subset)}
        for field in ("dice", "raw_dice", "assd_mm", "hd95_mm", "area_error_mm2"):
            record.update({f"{field}_{name}": value for name, value in describe([row[field] for row in subset]).items()})
        result.append(record)
    return result
