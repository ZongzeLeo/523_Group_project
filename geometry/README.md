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
