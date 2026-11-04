"""Audit the publisher's non-monotonic volume response to an apical cut."""
import argparse
import json
from pathlib import Path

import numpy as np

from data.dataset import CamusDataset
from data.download_camus import atomic_json, sha256
from data.protocol import identity, verify_results
from geometry.comparators import measure_mask
from geometry.publisher import measure_publisher, NOTEBOOK_SHA256
from geometry.publisher_profiles import publisher_profile, measure_profiles
from .perturbations import perturb_mask
from .run import read_csv


def audit(prepared, results):
    prepared, results = Path(prepared), Path(results)
    metadata = verify_results(results)
    if metadata["data_identity"] != identity(prepared):
        raise ValueError("Sampling audit inputs differ from the experiment")
    scenario = "apical_truncation_p2mm_both_ED"
    selected = [r for r in read_csv(results/"patients.csv") if r["method"] == "publisher_original" and r["scenario_id"] == scenario]
    positive = sorted([r for r in selected if float(r["delta_edv_ml"]) > 0], key=lambda r: r["patient_id"])
    if not positive:
        raise ValueError("This diagnostic requires an observed positive EDV response")
    example = positive[0]
    native = CamusDataset(prepared, "test").patient_frames(example["patient_id"])
    changed, areas = {}, {}
    for key, frame in native.items():
        changed[key] = frame
        if key.endswith("_ED"):
            spacing = frame["geometry"]["spacing_xy_mm"]
            gm = measure_mask(frame["mask"], spacing)
            mask, intervention = perturb_mask(frame["mask"], spacing, gm, "apical_truncation", 2.)
            changed[key] = dict(frame, mask=mask)
            areas[key] = intervention["area_change_mm2"]
    values = {}
    for label, frames in (("baseline", native), ("perturbed", changed)):
        profiles = {k: publisher_profile(v) for k, v in frames.items()}
        direct, cached = measure_publisher(frames), measure_profiles(profiles)
        for field in ("edv_ml", "esv_ml", "ef_pct"):
            if direct[field] != cached[field] or direct[field] != float(example[label+"_"+field]):
                raise ValueError("Diagnostic differs from the original notebook or saved scenario")
        step = max(p["step_mm"] for p in profiles.values())
        product = float(np.dot(profiles["2CH_ED"]["diameters_mm"], profiles["4CH_ED"]["diameters_mm"]))
        values[label] = {"shared_step_mm": step, "ed_diameter_product_sum_mm2": product,
                         "unrounded_edv_from_profiles_ml": product*step*np.pi/4/1000,
                         "original_rounded_edv_ml": direct["edv_ml"],
                         "original_rounded_esv_ml": direct["esv_ml"], "original_rounded_ef_pct": direct["ef_pct"]}
    result = {"status": "passed", "scenario_id": scenario, "patient_id": example["patient_id"],
              "selection": "First patient by ID with a positive EDV response; post hoc numerical diagnostic",
              "cohort": {"patients": len(selected), "edv_increased": len(positive),
                         "edv_unchanged": sum(float(r["delta_edv_ml"]) == 0 for r in selected),
                         "edv_decreased": sum(float(r["delta_edv_ml"]) < 0 for r in selected)},
              "area_changes_mm2": areas, **values,
              "step_ratio": values["perturbed"]["shared_step_mm"]/values["baseline"]["shared_step_mm"],
              "diameter_product_ratio": values["perturbed"]["ed_diameter_product_sum_mm2"]/values["baseline"]["ed_diameter_product_sum_mm2"],
              "interpretation": "The cut reduces native area but redetected axes and resampled disk widths can increase the geometric volume estimate. This is a property of the published estimator, not an increase in true cavity volume.",
              "source_results_metadata_sha256": sha256(results/"run_metadata.json"),
              "code_sha256": sha256(Path(__file__)), "publisher_notebook_sha256": NOTEBOOK_SHA256}
    atomic_json(results/"sampling_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--results", type=Path, default=Path("results/sensitivity_v1"))
    args = parser.parse_args()
    audit(args.prepared, args.results)
