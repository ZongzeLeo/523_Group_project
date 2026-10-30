# Data preparation

The workflow uses the CAMUS release containing 500 patients. Each patient
contributes two-chamber (2CH) and four-chamber (4CH) images at end-diastole (ED)
and end-systole (ES), together with expert masks, native pixel spacing, and
reference EF. LV cavity corresponds to label 1.

The publisher's split is retained at patient level:

| Partition | Patients | Frames |
|---|---:|---:|
| Training | 400 | 1,600 |
| Validation | 50 | 200 |
| Test | 50 | 200 |

Preparation verifies image–mask alignment, physical units, phase pairing,
nonempty cavity masks, consistent reference EF, and disjoint patient partitions.
Pixel-level duplicate detection found one identical image pair within training
and no cross-partition duplicates; the audit retains this finding.

Images are divided by 255 and resampled into a 256 × 256 grid using an isotropic,
physical-aspect-preserving letterbox. Images use linear interpolation and masks
use nearest-neighbour interpolation. The valid field of view is recorded for
loss computation. Native images, masks, spacing, and the inverse transform remain
available for volume estimation.

```bash
python -m data.download_camus
python -m data.prepare
python -m data.validate --check-reuse
```

Downloaded images and generated arrays are local outputs. The repository retains
the [split manifest](reference/splits.csv), [preprocessing specification](reference/preprocessing.json),
[source hashes](reference/source_hashes.json), and [data audit](reference/audit.json).
The available release does not provide independent clinical volume labels or
expert landmark coordinates.
