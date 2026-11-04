# Expert-mask geometry reproduction

The geometry stage executes the reviewed function cells of the publisher's
[original notebook](vendor/script_camus_ef.ipynb). A SHA-256 check ensures that
the numerical implementation is unchanged; demonstration and interactive cells
are not executed. Inputs consist solely of four native binary masks and their
pixel spacing.

The calculation resamples masks isotropically, extracts contours, identifies the
basal segment and apex, and measures 20 paired disk diameters. Its integration
uses a common disk height determined by the longest axis across both views and
phases. Reproducing this convention together with contour sampling, endpoint
handling, and rounding is necessary for agreement with the supplied labels.

The ED and ES volumes are rounded to integer millilitres before EF is computed:

$$\mathrm{EF}=\operatorname{round}\left[100\frac{\mathrm{EDV}-\mathrm{ESV}}{\mathrm{EDV}}\right].$$

On the 50 test patients, the implementation reproduced **all 50 rounded EF
labels exactly**, with **MAE 0.00 percentage points** and **zero failed patients**.
This establishes consistency with the CAMUS implementation. The resulting
volumes serve as mask-derived references for the subsequent segmentation study.

```bash
python -m geometry.reproduce_publisher
```

The [patient measurements](../results/geometry/patients.csv) and
[summary](../results/geometry/summary.json) retain every patient. Warnings from
degenerate candidate contour chords in the original routine are recorded; all
reported test volumes and EF values are finite and valid.

## Secondary geometric methods

The [controlled-error study](../results/sensitivity_v1/REPORT.md) also evaluates
two previously frozen mask-based methods in `comparators.py`. Both detect the
basal closure and farthest contour point on the native physical grid. They use
the longer view axis separately at ED and ES and retain continuous volumes/EF:

* Phase-specific Simpson: `V = pi/4 * max(L2,L4)/20 * sum(d2_i*d4_i)/1000`.
* Biplane ellipsoid area-length: `V = 8*A2*A4/(3*pi*min(L2,L4))/1000`.

Lengths and areas are in mm and mm squared; volumes are in mL. The area-length
variant is explicitly defined for paired apical views. These conventions differ
from the original publisher calculation. Their numerical implementation was
copied from the existing local comparison and checked against all 150 archived
patient-method results. See the [protocol](../docs/perturbation_protocol.md) for
landmark assumptions, topology checks, and interpretation.

`publisher_profiles.py` reuses the original notebook's per-frame outputs across
perturbation scenarios. It passes copies into the original volume function and
checks against complete original calls; numerical notebook functions are unchanged.
