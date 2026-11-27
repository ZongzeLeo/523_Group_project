# Reading the CAMUS patient results

The CSV contains measurements for **50 CAMUS test patients**. Each row is identified by `patient_id`, `condition`, `edited_views`, `edited_phases`, and `method`. Each patient has 60 rows, for a total of 3,000. [Open the patient results CSV](leo_patient_results.csv)

## Find a patient or condition

In Excel, select **Data → Filter** and choose a `patient_id`. To compare patients, clear that filter and select one `condition`, `method`, `edited_views`, and `edited_phases`. This gives one row per patient.

| `condition` | Masks and methods | Rows per patient | Total rows |
|---|---|---:|---:|
| `expert_control` | Original expert masks, measured with `camus_disk`, `area_length`, and `phase_specific_simpson` | 3 | 150 |
| `unet_prediction` | Sayeh’s post-processed U-Net masks, measured with the same three methods | 3 | 150 |
| `apical_cut_5mm`, `basal_inward_10mm`, `basal_outward_10mm` | Expert masks with one contour edit, measured with `camus_disk` and `area_length` | 54 | 2,700 |

Each edit was applied separately to the two-chamber view (`2CH`), four-chamber view (`4CH`), or `both`, at end-diastole (`ED`), end-systole (`ES`), or `both`. The three edits, three view choices, and three phase choices give 27 perturbations per patient for each of the two error-injection methods. For example, `edited_views = 2CH` and `edited_phases = ED` selects a cut or shift confined to the 2CH end-diastolic frame. Selecting `both` for both fields means all four frames were edited. Unedited rows use `none` for both fields. All four frames contribute to a complete biplane LVEF estimate.

## Choose the reference for a difference

`edv_ml` and `esv_ml` are the estimated end-diastolic and end-systolic volumes in mL. `lvef_pct` is the left ventricular ejection fraction in percent. LVEF differences are expressed in **percentage points (pp)**. A positive signed difference means the row estimate exceeds its reference.

| Question | Columns to use | Reference and sign |
|---|---|---|
| How much did the prediction or contour edit change the estimate? | `expert_control_*`, `delta_*_vs_expert_control_*`, `abs_delta_*` | Row estimate minus the same method on the same patient’s unmodified expert masks. The `abs_` column gives the magnitude. |
| How close was LVEF to the supplied CAMUS label? | `camus_label_lvef_pct`, `lvef_minus_camus_label_pp`, `abs_lvef_minus_camus_label_pp` | Row LVEF minus the patient’s rounded CAMUS label. |
| How much did the methods differ on identical masks? | `al_minus_camus_disk_*`, `al_minus_phase_simpson_*` | Area–length minus the named disk method for the same patient and masks. These fields appear on `area_length` rows. The phase-specific comparison is available for controls and predictions with complete paired measurements. |

**Worked example.** Filter `patient_id = patient0027` and `method = area_length`. The two reference columns answer different questions even for the same prediction.

| Condition | Edited frames | LVEF (%) | Change from expert area–length (pp) | Difference from CAMUS label (pp) |
|---|---|---:|---:|---:|
| `expert_control` | None | 51.23 | 0.00 | +11.23 |
| `unet_prediction` | None | 52.03 | +0.80 | +12.03 |
| `basal_inward_10mm` | Both views and both phases | 53.18 | +1.95 | +13.18 |

The patient’s expert area–length estimate was 51.23% and the supplied CAMUS label was 40%. Values above are rounded for display.

## Reproduce the report summaries

Start each comparison with all rows visible, then apply the filters below across the full cohort. The table and figure numbers refer to the shared course report, `report_version2.docx`.

| Report item | Filters and calculation | LVEF check |
|---|---|---|
| **Table 5, area–length versus CAMUS disk** | Select `expert_control` and `area_length`. Average the signed `al_minus_camus_disk_*` differences for bias. For mean absolute error (MAE), take each patient’s absolute difference before averaging. | Bias +7.32 pp, MAE 7.32 pp |
| **Table 5, U-Net versus expert masks** | Select `unet_prediction` and `area_length`. Average `delta_*_vs_expert_control_*` for bias and the corresponding `abs_delta_*` for MAE. | Bias −1.35 pp, MAE 5.12 pp |
| **Figure 3, all perturbations** | Select the three contour-edit conditions. Group by `condition`, `method`, `edited_views`, and `edited_phases`. Average `delta_lvef_vs_expert_control_pp` in each group. | 54 cells, 50 patients per cell |
| **Table 6, all four frames edited** | Use the Figure 3 selection and set both scope fields to `both`. For each edit and method, average the signed and absolute changes in EDV, ESV, and LVEF. Parentheses show the mean absolute change. | Area–length with an inward shift gives +1.95 pp (mean absolute change 2.09 pp). |

In Excel, `=SUBTOTAL(101,Q2:Q3001)` averages the visible signed LVEF changes after filtering. Column `T` contains the corresponding absolute changes. Table 6 gives 95% percentile confidence intervals for the **mean signed LVEF change**, using 2,000 patient-level bootstrap resamples within each group (patient ID order, seed 523).

## Measurement details and quality flags

- `mask_source` distinguishes `expert` from `sayeh_unet_e80_postprocessed`. Both sources use label 1 for the LV cavity. `apical_cut_mm` records the cut depth. In `basal_shift_mm`, +10 means inward towards the apex and −10 means outward. `basal_taper_mm` gives the 30 mm distance towards the apex over which basal displacement decreases to zero.
- `measurement_landmarks` is `camus_publisher` for CAMUS disk and area–length, and `phase_specific_detector` for phase-specific Simpson. The `al_*_area_mm2` and `al_*_length_mm` fields give the cavity area and long-axis length for each view and phase on area–length rows. The volume formula uses the shorter of the two view lengths at each phase.
- `camus_disk` uses the publisher’s shared disk height across all four frames. Its `lvef_pct` includes the final rounding to a whole percentage point. `camus_disk_lvef_before_final_rounding_pct` gives LVEF calculated from the rounded disk volumes before that last step.
- For report summaries, include `measurement_status` values `ok` and `negative_lvef_retained`. Four single-phase basal shifts in `patient0052` gave area–length ESV greater than EDV. The reported means include those four negative LVEF measurements.
- `incomplete_geometry` occurs once, for the U-Net phase-specific Simpson measurement in `patient0208`. Its ESV is available, while EDV, LVEF, and paired changes are blank. `failed_frames_and_reason` identifies the failed frame. `mask_qc` records frame-level observations such as contact with the image border or a mask hole. An empty numeric cell means unavailable or inapplicable. In a difference column, zero means the paired values were equal.
