"""Write a concise academic report from verified sensitivity result tables."""
import argparse
import json
from pathlib import Path

from data.download_camus import sha256
from data.protocol import verify_results
from .run import read_csv
from .sensitivity_figures import METHOD_NAMES, FEATURE_LABELS
from .perturbations import METHODS, FRAME_KEYS


def f(value, digits=2, signed=False):
    if value in (None, ""):
        return "NA"
    return format(float(value), ("+" if signed else "")+f".{digits}f")


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def generate(results):
    results=Path(results)
    verify_results(results)
    summary=json.loads((results/"summary.json").read_text())
    sensitivity=read_csv(results/"sensitivity_summary.csv")
    patients=read_csv(results/"observed_patients.csv")
    comparisons=read_csv(results/"observed_comparison.csv")
    associations=read_csv(results/"associations.csv")
    primary=[r for r in patients if r["method"]=="publisher_original"]
    audits=read_csv(results/"perturbation_frames.csv")
    failures=[r for r in audits if r["geometry_status"]!="ok"]
    sampling=json.loads((results/"sampling_audit.json").read_text())
    effects_index={r['scenario_id']:r for r in sensitivity if r['method']=='publisher_original' and r['group_type']=='overall'}
    area_association=next(r for r in associations if r['method']=='publisher_original' and r['group_type']=='overall'
                          and r['view_phase']=='2CH_ES' and r['feature']=='area_error_pct')
    failure_text=table(['Frame','Intervention (mm)','Rejection reason'],
                       [[r['frame_id'],f(r['magnitude_mm'],signed=True),r['geometry_error']] for r in failures])
    baseline=[]
    for method in METHODS:
        c=next(r for r in comparisons if r["method"]==method and r["group_type"]=="overall")
        selected=[r for r in patients if r["method"]==method]
        expert_mae=sum(abs(float(r["gt_ef_pct"])-float(r["camus_reference_ef_pct"])) for r in selected)/len(selected)
        baseline.append([METHOD_NAMES[method],f(expert_mae),f(c["ef_vs_gt_pp_mae"]),f(c["ef_vs_camus_pp_mae"]),
                         f(c["edv_vs_gt_ml_mae"]),f(c["esv_vs_gt_ml_mae"])])
    effects=[]
    for family,value,label in (("apical_truncation",2.,"Apical cut 2 mm"),("basal_displacement",2.,"Basal inward 2 mm"),
                               ("basal_displacement",-2.,"Basal outward 2 mm")):
        for phase in ("ED","ES","both"):
            r=next(r for r in sensitivity if r["method"]=="publisher_original" and r["group_type"]=="overall"
                   and r["family"]==family and float(r["magnitude_mm"])==value and r["view_scope"]=="both" and r["phase_scope"]==phase)
            ci=f"{f(r['ef_pp_mean'],signed=True)} [{f(r['ef_pp_mean_ci95_low'])}, {f(r['ef_pp_mean_ci95_high'])}]"
            effects.append([label,phase if phase!="both" else "ED + ES",f(r["edv_ml_mean"],signed=True),
                            f(r["esv_ml_mean"],signed=True),ci,f(r["ef_pp_mae"])])
    association_rows=[]
    for feature,label in FEATURE_LABELS.items():
        values=[]
        for key in FRAME_KEYS:
            r=next(r for r in associations if r["method"]=="publisher_original" and r["group_type"]=="overall"
                   and r["feature"]==feature and r["view_phase"]==key)
            values.append(f(r["spearman_rho"]))
        association_rows.append([label,*values])
    strata=[]
    for group_type in ("reference_ef_group","image_quality"):
        for group in sorted({r["group"] for r in sensitivity if r["group_type"]==group_type}):
            values=[]
            for family,value in (("apical_truncation",2.),("basal_displacement",2.),("basal_displacement",-2.)):
                r=next(r for r in sensitivity if r["method"]=="publisher_original" and r["group_type"]==group_type and r["group"]==group
                       and r["family"]==family and float(r["magnitude_mm"])==value and r["view_scope"]=="both" and r["phase_scope"]=="ES")
                values.append(f(r["ef_pp_mae"]))
            label={"le45":"EF <=45%","gt45_lt55":"EF >45% to <55%","ge55":"EF >=55%"}.get(group,group)
            strata.append([label,r["attempted_patients"],*values])
    outliers=sorted(primary,key=lambda r:(-abs(float(r["ef_error_vs_gt_pp"])),r["patient_id"]))[:3]
    cases=[[r["patient_id"],f(r["mean_frame_dice"],3),f(r["ef_error_vs_gt_pp"],signed=True),
            f(r["edv_contribution_pp"],signed=True),f(r["esv_contribution_pp"],signed=True),
            f(r["ef_rounding_remainder_pp"],signed=True)] for r in outliers]
    text=f"""# Controlled contour errors and LV functional estimation

## Study design

This analysis completes the proposal's controlled error-propagation experiment.
The original 50 test patients, 200 native frames, and selected GroupNorm U-Net
remain fixed. Apical truncation (1 and 2 mm) and basal displacement (-2, -1, +1,
and +2 mm) were independently applied to expert masks in 2CH, 4CH, or both views
at ED, ES, or both phases. Positive basal displacement moves the boundary toward
the apex within a 10-mm tapering band. Interventions use native physical spacing
and a coordinate system derived from the original expert mask; geometry is
redetected after modification. The [protocol](../../docs/perturbation_protocol.md)
defines the deformation, rasterization, and all comparisons.

The 54 nonzero combinations and one shared control produce **8,250
patient-scenario-method evaluations** across three methods. Each change is
measured against the same patient's unperturbed mask under the same method.
The publisher calculation retains the original functions and rounding;
phase-specific Simpson and the explicitly defined biplane ellipsoid area-length
variant retain continuous measurements. Their comparison includes differences
in landmark detection, disk-height convention, sampling, and rounding, and
therefore is not an isolated comparison of integration formulas.

![Physical interventions and redetected landmarks](figures/perturbation_overlays.png)

**Figure 1.** Native contours and mask-derived long axes for the patient closest
to median baseline patient-mean Dice. Each row applies its intervention
independently to the original mask. Basal midpoint and apex are marked by squares
and circles. Nominal displacement and measured landmark movement are distinct:
native rasterization and landmark redetection can alter the observed response.

## Baseline and geometric-method comparison

All **{summary['publisher_expert_exact_matches']}/50** unchanged expert-mask EF
values reproduce the supplied rounded CAMUS labels. The observed segmentation
and geometric measurements reproduce the previous evaluation. Expert volume
references are derived from masks using the corresponding method.

{table(['Method','Expert EF MAE vs CAMUS (pp)','Predicted EF MAE vs expert masks (pp)','Predicted EF MAE vs CAMUS (pp)','EDV MAE vs expert masks (mL)','ESV MAE vs expert masks (mL)'],baseline)}

The publisher result establishes computational agreement with CAMUS's rounded
labels. The secondary methods use different conventions; their direct CAMUS
errors include this baseline difference. Within-method predicted-versus-expert
comparisons measure the effect of replacing the expert contour with a prediction.

## Controlled sensitivity

Across all evaluations, **{summary['failed_geometry_rows']} geometry failures**,
**{summary['invalid_volume_rows']} invalid-volume results**, and
**{summary['border_contact_rows']} border-contact results** were recorded. These
are counts of scenario-method evaluations, not independent patients. All 150
unchanged patient-method controls yielded zero volume and EF changes.
The table below reports the primary publisher calculation for perturbations
applied to both views. Signed changes are perturbed minus unperturbed; confidence
intervals use 2,000 paired patient bootstrap resamples. The complete view-specific,
method-specific, and subgroup estimates are retained in the result tables.

{table(['Intervention','Affected phase','Mean delta EDV (mL)','Mean delta ESV (mL)','Mean delta EF [95% CI] (pp)','Mean absolute delta EF (pp)'],effects)}

Apical truncation in both views produced opposite signed responses when applied
at ED and ES: the 2-mm intervention changed mean EF by
**{f(effects_index['apical_truncation_p2mm_both_ED']['ef_pp_mean'],signed=True)} pp**
and **{f(effects_index['apical_truncation_p2mm_both_ES']['ef_pp_mean'],signed=True)} pp**,
respectively. Applying the same cut at both phases reduced the mean signed change
to **{f(effects_index['apical_truncation_p2mm_both_both']['ef_pp_mean'],signed=True)} pp**,
while the mean absolute change remained
**{f(effects_index['apical_truncation_p2mm_both_both']['ef_pp_mae'])} pp**. Thus,
similar contour errors across phases can partially cancel in the EF ratio.

The estimated volume response need not follow the removed mask area. For the
2-mm cut in both ED views, estimated EDV increased in
{sampling['cohort']['edv_increased']}/50 patients, was unchanged in
{sampling['cohort']['edv_unchanged']}, and decreased in {sampling['cohort']['edv_decreased']}.
A [post hoc numerical audit](sampling_audit.json) of {sampling['patient_id']}
reproduced this response with the complete original notebook: disk height fell
by {f(100*(1-sampling['step_ratio']))}%, while the sum of paired disk-diameter
products rose by {f(100*(sampling['diameter_product_ratio']-1))}%, changing rounded
EDV from {sampling['baseline']['original_rounded_edv_ml']} to
{sampling['perturbed']['original_rounded_edv_ml']} mL. This reflects redetection
and disk sampling in the estimator, not an increase in true cavity volume.

![Signed EF sensitivity and patient-bootstrap intervals](figures/sensitivity_curves.png)

**Figure 2.** Signed EF responses to predefined physical perturbations under the
publisher calculation. Shading denotes 95% confidence intervals for the mean
paired response. Lines connect the tested magnitudes and do not imply a fitted
continuous response. The four-frame shared disk height can affect both volume
estimates after a single-phase intervention; integer rounding can also produce
plateaus or step changes.

![Mean absolute sensitivity across methods, views, and phases](figures/method_sensitivity.png)

**Figure 3.** Mean absolute within-method EF change. A common color scale supports
comparison across all methods. Absolute changes reveal responses that can cancel
in the signed mean. Means use available numeric pairs, with reduced sample sizes
marked in the affected cells. Paired differences between methods, with patient-bootstrap
intervals, are provided in [method contrasts](method_contrasts.csv).

The 16 rejected scenario-method evaluations originate from two single-pixel
interior holes introduced by nearest-neighbour basal deformation. Each affected
frame participates in four scenarios and both secondary methods. These are
digital rasterization failures under the fixed topology check, not evidence of
anatomical instability. The publisher method does not apply that rejection rule
and returned finite estimates in all 2,750 scenarios. Complete-case method
contrasts retain their paired sample counts; failures are never converted to
zero errors or silently repaired.

{failure_text}

![Rasterization failures retained by the fixed protocol](figures/rasterization_failures.png)

**Failure audit.** Native contours and enlarged masks identify the two interior
background pixels. The main experiment retains the uncorrected masks.

For descriptive subgroup inspection, the following table fixes the intervention
at 2 mm in both ES views. Patient-level image quality is the poorer of the two
views. Each entry is mean absolute EF change in percentage points; confidence
intervals and all other scenarios are available in the complete summary.

{table(['Stratum','Patients','Apical cut 2 mm','Basal inward 2 mm','Basal outward 2 mm'],strata)}

## Observed segmentation errors

All prespecified associations use one observation per patient within each
view/phase combination. The first two rows below relate signed contour errors
to signed EF errors; remaining rows relate geometric discrepancy to absolute
EF error. Values are Spearman rank correlations for the publisher calculation.
All three methods and strata, including bootstrap intervals and undefined
estimates, appear in [the association table](associations.csv).

{table(['Frame-level discrepancy',*FRAME_KEYS],association_rows)}

In 2CH ES, area error was associated with signed EF error
(rho **{f(area_association['spearman_rho'])}**, 95% CI
**{f(area_association['ci95_low'])} to {f(area_association['ci95_high'])}**).
The negative direction is consistent with larger estimated ES cavities producing
lower EF, although paired-view and ED errors also contribute. The weaker
landmark associations show that a single displacement measure does not capture
the full error pattern.

![Observed error association matrix](figures/association_matrix.png)

**Figure 4.** All 84 unstratified prespecified associations. Coefficients are
descriptive, with no significance-based selection or multiplicity-adjusted
confirmatory claim. Area and axis errors are signed in the first two rows;
the remaining features describe discrepancy magnitude.

![Observed 2CH ES errors and EF deviation](figures/observed_error_associations.png)

**Figure 5.** Patient-level associations for 2CH ES, the lowest-Dice view/phase in
the completed baseline. The intervals resample patients and rerank ties within
each replicate. Association does not establish that a particular anatomical
error caused the EF deviation.

The largest baseline EF errors are retained below. An exact arithmetic
decomposition separates the EDV and ESV contributions, with a remainder for
final EF rounding. With expert volumes D and E and predicted volumes Dp and Ep,
the terms are `100*E*(Dp-D)/(D*Dp)` and `-100*(Ep-E)/Dp`. Their sum equals the
continuous EF difference; adding the recorded rounding remainder recovers the
reported EF error.

{table(['Patient','Mean Dice','EF error (pp)','EDV term (pp)','ESV term (pp)','Rounding (pp)'],cases)}

The ESV term has the largest magnitude in each of these three cases. In
patient0241, a mean Dice of 0.944 coexists with a +16-pp EF error, of which the
ESV term contributes +12.35 pp. Overlap accuracy alone therefore does not fully
describe functional agreement in this test set.

![Largest-error anatomy and exact EF contribution terms](figures/largest_error_anatomy.png)

**Figure 6.** Expert and predicted contours with the independent mask detector's
landmark surrogates. These landmarks describe contour discrepancies and are not
the publisher's separate landmark calculation. The contribution terms explain
the EF arithmetic, not the anatomical cause of the volume errors.

## Interpretation and reproducibility

The experiment separates controlled contour sensitivity from the observed model
errors while preserving each method's baseline. It supports evaluation of which
regions, phases, and views affect the reported functional estimate. Absolute and
signed responses should be interpreted together because EDV and ESV errors can
reinforce or offset one another. Differences among methods reflect their full
declared computational conventions.

This is a descriptive analysis of one model and one 50-patient internal test
partition. The protocol was fixed after inspecting the baseline, and the
perturbations are simplified synthetic interventions. Small EF/quality strata
and multiple exploratory associations limit inference. Independent clinical
EDV/ESV and anatomical landmark annotations are unavailable. Zero expert-mask
EF error therefore establishes numerical reproduction, not independent clinical
accuracy. No test outcome was used to retrain the model or tune the protocol.

The package includes [patient/scenario results](patients.csv),
[frame intervention audits](perturbation_frames.csv),
[paired sensitivity summaries](sensitivity_summary.csv),
[observed frame measurements](observed_frames.csv), and
[observed patient measurements](observed_patients.csv).
The independent [validation receipt](validation.json) checks patient pairing,
coverage, unchanged controls, EF arithmetic, summary means, method contrasts,
rank correlations, and reproduction of the archived baseline.
See [reproduction commands](../../docs/reproduction.md#controlled-error-analysis).
"""
    (results/"REPORT.md").write_bytes(text.encode("utf-8"))
    print(json.dumps({"report":str(results/"REPORT.md"),"sha256":sha256(results/"REPORT.md")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=Path("results/sensitivity_v1"))
    args=parser.parse_args()
    generate(args.results)
