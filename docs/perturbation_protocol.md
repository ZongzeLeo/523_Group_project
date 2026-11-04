# Controlled contour errors and functional sensitivity

This fixed descriptive protocol completes the error-propagation experiment in
the project proposal. It uses the existing 50-patient test partition and selected
GroupNorm model without retraining, threshold adjustment, or outcome-based
selection of perturbation magnitudes. The protocol is specified after the
baseline results were inspected; it is not a prospectively registered study.

## Interventions

Each native expert LV mask defines a basal midpoint and apex through the frozen
mask-based detector. Their physical coordinates define an axis with coordinate
`z=0` at the base and `z=L` at the apex. All distances use native pixel spacing.

* **Apical truncation:** remove mask pixels with `z > L-d`, for `d=1,2 mm`.
* **Basal displacement:** move points along the axis by `d*w(z)`, for
  `d=-2,-1,+1,+2 mm`. Positive values move the basal region inward toward the
  apex; negative values extend it outward. The weight is one for `z<=0`,
  decreases linearly to zero over `0<z<10 mm`, and is zero above 10 mm.
  The analytic inverse deformation is sampled on the original grid using
  nearest-neighbour interpolation. The region above 10 mm is unchanged.
* **Control:** return an independent, bitwise-identical copy of every mask.

These are explicit synthetic contour errors, not models of observer behaviour.
The apical cut introduces a flat cap; the basal operation deforms a local band
rather than translating the entire cavity. Native rasterization can make the
achieved displacement differ from its nominal value. Pixel changes, physical
area changes, and projected apical/basal extent changes are recorded.

For each of the six nonzero interventions, perturb 2CH only, 4CH only, or both
views, crossed with ED only, ES only, or both phases. The 54 nonzero combinations
and one shared control yield 55 scenarios per patient. Pairing never crosses
patients. Interventions are independent applications to the original mask,
not sequential accumulations. No mask cleanup is applied. Detected landmarks
are recalculated for volume estimation after each intervention; the original
landmarks define only the intervention coordinate system.

## Geometric comparisons

The primary method is the unchanged, hash-pinned CAMUS publisher calculation,
including its shared four-frame disk height and integer volume/EF rounding.
Secondary methods reuse the previously frozen local implementation, without
numerical changes: phase-specific 20-disk Simpson biplane integration and the
explicit biplane ellipsoid area-length formula
`V = 8*A2*A4/(3*pi*min(L2,L4))/1000` in mL for mm-based inputs.
The secondary methods retain continuous volumes and EF. They are conventionally
different from the publisher calculation; their CAMUS-label offsets are not
calibrated away. The area-length variant uses the two apical views and does not
assume a parasternal short-axis image.

Repeated scenarios reuse the original notebook's per-frame diameter profiles.
The original volume function receives copies of those arrays because it converts
their units in place. Its numerical functions are unchanged. The cached result
is checked against the complete original call for every unchanged patient;
additional perturbed cases and analytic masks check numerical equivalence.

For every patient, method, and scenario, compute perturbed-minus-unperturbed
EDV, ESV, and EF using that same method. Primary comparisons are paired within
patient. Report mean signed change, mean absolute change, 95% percentile
bootstrap intervals (2,000 patient resamples; seed 523), and failure counts.
Invalid-volume results remain explicitly flagged; finite negative EF is retained.
Missing geometry never becomes a zero error. One zero-control row is retained
per patient/method; repeated measurements are not independent samples.

## Observed prediction errors

Remeasure all 200 saved native predicted masks with the frozen secondary detector
and compare with the corresponding expert mask. Examine area, long-axis length,
apex, and basal-midpoint discrepancies separately for each of four view/phase
combinations. Each association therefore uses at most one observation per
patient. Report signed-area/length associations with signed EF error and
absolute geometric discrepancies with absolute EF error using Spearman rho
and patient-bootstrap intervals. These are exploratory associations without
causal interpretation or significance-based selection. All prespecified
associations, their sample sizes, and undefined estimates are retained.

Summaries additionally report the predefined reference-EF groups (<=45%,
>45% to <55%, >=55%) and image quality. Patient-level quality is the worse
of the two view qualities. Small strata remain descriptive. Landmark differences
compare mask-derived surrogates; independently annotated anatomical landmarks
and independent clinical EDV/ESV measurements are unavailable.

## Validation and figures

Check physical displacement and locality on analytic masks, the 55-scenario
matrix, unchanged controls, paired patient identities, artifact hashes, and
reproduction of all archived baseline measurements. Visual review includes
native perturbation/landmark overlays, sensitivity curves, a view/phase heatmap,
observed-error association plots, and the largest original EF-error cases.
The primary publisher's unchanged-mask expert EF must reproduce all 50 supplied
rounded labels. Results and failures remain in inspectable CSV tables.
