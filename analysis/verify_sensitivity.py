"""Independent stdlib checks of the frozen experiment and its result tables."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import statistics

from .verify import digest, read_csv, require


def numeric(value):
    return None if value in (None, "") else float(value)


def near(actual, expected, label):
    actual = numeric(actual)
    if actual is None or expected is None:
        require(actual is None and expected is None, "Missing-value mismatch: "+label)
    else:
        require(math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-10), "Mismatch: "+label)


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.]*len(values)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        for j in range(start, end):
            result[ordered[j]] = (start+end-1)/2+1
        start = end
    return result


def rho(x, y):
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    a, b = ranks(x), ranks(y)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((i-ma)*(j-mb) for i, j in zip(a, b))
    return numerator/math.sqrt(sum((i-ma)**2 for i in a)*sum((j-mb)**2 for j in b))


def selected(records, group_type, group):
    return records if group_type == "overall" else [r for r in records if r[group_type] == group]


def verify(root, results=None, archive=None):
    root = Path(root)
    folder = Path(results) if results else root/"results/sensitivity_v1"
    config = json.loads((folder/"config.json").read_text())
    metadata = json.loads((folder/"run_metadata.json").read_text())
    require(metadata["status"] == "complete", "Incomplete sensitivity run")
    require(metadata["protocol_sha256"] == digest(root/"analysis/configs/perturbations_v1.json"), "Protocol changed")
    require(config == json.loads((root/"analysis/configs/perturbations_v1.json").read_text()), "Saved protocol mismatch")
    for name, sha in metadata["artifact_sha256"].items():
        require(digest(folder/name) == sha, "Sensitivity artifact changed: "+name)
    for name, sha in metadata["code_sha256"].items():
        require(digest(root/name) == sha, "Sensitivity source changed: "+name)
    ids = {r["patient_id"] for r in read_csv(root/"data/reference/splits.csv") if r["split"] == "test"}
    require(len(ids) == 50 and set(metadata["patients"]) == ids, "Patient split differs")
    checkpoint = json.loads((root/"results/prediction_metadata.json").read_text())["checkpoint_sha256"]
    require(metadata["checkpoint_sha256"] == checkpoint, "Selected model changed")
    matrix = read_csv(folder/"scenarios.csv")
    expected_matrix = {("control", 0., "none", "none")}
    for family, magnitudes in (("apical_truncation", config["apical_truncation_mm"]), ("basal_displacement", config["basal_displacement_mm"])):
        expected_matrix |= {(family, float(d), v, p) for d, v, p in product(magnitudes, ("2CH", "4CH", "both"), ("ED", "ES", "both"))}
    require({(r["family"], float(r["magnitude_mm"]), r["view_scope"], r["phase_scope"]) for r in matrix} == expected_matrix,
            "Incomplete intervention matrix")
    require(len(matrix) == len(expected_matrix) == 55, "Duplicate or missing scenarios")
    scenario_map = {r["scenario_id"]: r for r in matrix}
    rows = read_csv(folder/"patients.csv")
    index = {(r["patient_id"], r["scenario_id"], r["method"]): r for r in rows}
    require(len(rows) == len(index) == 8250, "Duplicate or missing patient/method/scenario rows")
    require(set(index) == set(product(ids, scenario_map, config["methods"])), "Incomplete paired experiment")
    by_scenario = defaultdict(list)
    for r in rows:
        baseline = index[r["patient_id"], "control", r["method"]]
        declared = scenario_map[r["scenario_id"]]
        require(all(r[k] == declared[k] for k in ("family", "magnitude_mm", "view_scope", "phase_scope")), "Scenario metadata mismatch")
        for field, quantity in (("edv_ml", "edv_ml"), ("esv_ml", "esv_ml"), ("ef_pct", "ef_pp")):
            near(r["baseline_"+field], numeric(baseline["perturbed_"+field]), "paired baseline")
            value, base = numeric(r["perturbed_"+field]), numeric(r["baseline_"+field])
            near(r["delta_"+quantity], value-base if value is not None else None, "paired delta")
            if r["scenario_id"] == "control":
                near(r["delta_"+quantity], 0., "zero control")
        edv, esv, ef = (numeric(r["perturbed_"+k]) for k in ("edv_ml", "esv_ml", "ef_pct"))
        if ef is None:
            require(r["status"] == "invalid_geometry", "Uncounted missing geometry")
        else:
            require(edv > 0 and esv >= 0, "Invalid volume escaped validation")
            expected = 100*(edv-esv)/edv
            near(ef, round(expected) if r["method"] == "publisher_original" else expected, "EF formula")
            require(r["status"] == ("invalid_volumes" if esv > edv else "ok"), "Invalid-volume flag mismatch")
        if r["method"] != "publisher_original":
            for phase, quantity in (("ED", "esv_ml"), ("ES", "edv_ml")):
                if r["phase_scope"] == phase and r["status"] != "invalid_geometry":
                    near(r["delta_"+quantity], 0., "Unperturbed secondary phase")
        by_scenario[r["method"], r["scenario_id"]].append(r)
    summaries = read_csv(folder/"sensitivity_summary.csv")
    require(len(summaries) == 1155, "Missing sensitivity strata")
    for s in summaries:
        subset = selected(by_scenario[s["method"], s["scenario_id"]], s["group_type"], s["group"])
        near(s["attempted_patients"], len(subset), "stratum count")
        for quantity in ("edv_ml", "esv_ml", "ef_pp"):
            values = [numeric(r["delta_"+quantity]) for r in subset]
            values = [v for v in values if v is not None]
            near(s[quantity+"_numeric_pairs"], len(values), "numeric pair coverage")
            near(s[quantity+"_mean"], statistics.mean(values) if values else None, "mean signed change")
            near(s[quantity+"_mae"], statistics.mean(map(abs, values)) if values else None, "mean absolute change")
        for status, field in (("invalid_geometry", "failed_geometry_patients"), ("invalid_volumes", "invalid_volume_patients")):
            near(s[field], sum(r["status"] == status for r in subset), "failure coverage")
    contrasts = read_csv(folder/"method_contrasts.csv")
    require(len(contrasts) == 495, "Missing paired method contrasts")
    for r in contrasts:
        pairs = [(numeric(index[p, r["scenario_id"], r["method_a"]]["delta_"+r["quantity"]]),
                  numeric(index[p, r["scenario_id"], r["method_b"]]["delta_"+r["quantity"]])) for p in ids]
        delta = [a-b for a, b in pairs if a is not None and b is not None]
        near(r["numeric_pairs"], len(delta), "method contrast coverage")
        near(r["mean_delta_a_minus_b"], statistics.mean(delta) if delta else None, "paired method contrast")
    audits = read_csv(folder/"perturbation_frames.csv")
    require(len(audits) == len({(r["frame_id"], r["family"], r["magnitude_mm"]) for r in audits}) == 1400,
            "Missing frame-variant audits")
    require(all(float(r["changed_pixels"]) == 0 and float(r["area_change_mm2"]) == 0 for r in audits if r["family"] == "control"),
            "Control masks changed")
    frames = read_csv(folder/"observed_frames.csv")
    observed = read_csv(folder/"observed_patients.csv")
    observed_index = {(r["patient_id"], r["method"]): r for r in observed}
    require(len(frames) == len({r["frame_id"] for r in frames}) == 200, "Observed frame coverage")
    require(len(observed) == len(observed_index) == 150, "Observed method coverage")
    for r in observed:
        if r["predicted_ef_pct"]:
            de = float(r["predicted_ef_pct"])-float(r["gt_ef_pct"])
            near(r["ef_error_vs_gt_pp"], de, "observed EF error")
            near(sum(float(r[k]) for k in ("edv_contribution_pp", "esv_contribution_pp", "ef_rounding_remainder_pp")), de,
                 "EF contribution identity")
    associations = read_csv(folder/"associations.csv")
    require(len(associations) == 588, "Missing prespecified associations")
    for r in associations:
        subset = [f for f in frames if f["view_phase"] == r["view_phase"]]
        subset = selected(subset, r["group_type"], r["group"])
        pairs = []
        for frame in subset:
            patient = observed_index[frame["patient_id"], r["method"]]
            x, y = numeric(frame[r["feature"]]), numeric(patient["ef_error_vs_gt_pp"])
            if x is not None and y is not None:
                pairs.append((x, abs(y) if r["target"] == "absolute_ef_error_pp" else y))
        near(r["n"], len(pairs), "association coverage")
        near(r["spearman_rho"], rho([a for a, b in pairs], [b for a, b in pairs]), "Spearman rho")
    original = {r["patient_id"]: r for r in read_csv(root/"results/analysis/patients.csv")}
    for pid, previous in original.items():
        current = observed_index[pid, "publisher_original"]
        for field in ("gt_edv_ml", "gt_esv_ml", "gt_ef_pct", "predicted_edv_ml", "predicted_esv_ml", "predicted_ef_pct"):
            near(current[field], float(previous[field]), "published baseline reproduction")
    archive_receipt = None
    if archive is not None:
        archive = Path(archive)
        previous = read_csv(archive/"patients.csv")
        require(len(previous) == 150, "Archived comparison incomplete")
        for r in previous:
            current = observed_index[r["patient_id"], r["method"]]
            for field in ("gt_edv_ml", "gt_esv_ml", "gt_ef_pct", "predicted_edv_ml", "predicted_esv_ml", "predicted_ef_pct"):
                near(current[field], float(r[field]), "archived method reproduction")
        frame_index = {r["frame_id"]: r for r in frames}
        for r in read_csv(archive/"frames.csv"):
            current = frame_index[r["frame_id"]]
            for field in ("dice", "area_error_mm2", "length_error_mm", "apex_displacement_mm", "base_midpoint_displacement_mm"):
                near(current[field], float(r[field]), "archived frame reproduction")
        archive_receipt = {"patient_rows_matched": 150, "frame_rows_matched": 200,
                           "source_sha256": {n: digest(archive/n) for n in ("patients.csv", "frames.csv")}}
    checked_figures = 0
    figure_manifest = folder/"figures/figure_manifest.json"
    if figure_manifest.exists():
        figures = json.loads(figure_manifest.read_text())
        require(figures["source_results_metadata_sha256"] == digest(folder/"run_metadata.json"), "Figure source mismatch")
        require(figures["plotting_code_sha256"] == digest(root/"analysis/sensitivity_figures.py"), "Plotting source changed")
        for name, sha in figures["artifact_sha256"].items():
            require(digest(folder/"figures"/name) == sha, "Sensitivity figure changed: "+name)
            checked_figures += 1
    if (folder/"sampling_audit.json").exists():
        audit = json.loads((folder/"sampling_audit.json").read_text())
        require(audit["source_results_metadata_sha256"] == digest(folder/"run_metadata.json"), "Sampling audit source mismatch")
        require(audit["code_sha256"] == digest(root/"analysis/sampling_audit.py"), "Sampling audit code changed")
        pair = index[audit["patient_id"], audit["scenario_id"], "publisher_original"]
        for which in ("baseline", "perturbed"):
            near(audit[which]["original_rounded_edv_ml"], float(pair[which+"_edv_ml"]), "Sampling diagnostic volume")
        near(audit["step_ratio"]*audit["diameter_product_ratio"],
             audit["perturbed"]["unrounded_edv_from_profiles_ml"]/audit["baseline"]["unrounded_edv_from_profiles_ml"], "Sampling factorization")
    return {"status": "passed", "patients": len(ids), "scenarios": len(matrix), "methods": 3,
            "patient_scenario_method_rows": len(rows), "frame_variant_rows": len(audits),
            "zero_control_rows": 150, "sensitivity_summary_rows_checked": len(summaries),
            "paired_method_contrasts_checked": len(contrasts), "associations_independently_recalculated": len(associations),
            "published_baseline_patients_matched": 50, "archived_comparison": archive_receipt,
            "figure_files_checked": checked_figures,
            "verification_code_sha256": digest(Path(__file__)), "checked_utc": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = verify(args.root, args.results, args.archive)
    text = json.dumps(receipt, indent=2)+"\n"
    if args.receipt:
        args.receipt.write_bytes(text.encode("utf-8"))
    print(text)
