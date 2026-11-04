# Reproduction

Run commands from the repository root. Raw images and generated arrays remain
local; the repository includes the selected checkpoint, immutable reference
manifests, patient measurements, and publication figures.

## Environment

Use Python 3.10–3.12 and PyTorch 2.5.1. Install a build appropriate to the available
hardware using the [official version-specific instructions](https://pytorch.org/get-started/previous-versions/#v251).
For the recorded CUDA 11.8 configuration:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-training.txt
```

The recorded training and inference environment used Python 3.10.19, PyTorch
2.5.1, CUDA 11.8, NumPy 2.0.1, SciPy 1.15.3, and an NVIDIA RTX 4090.
Data preparation and the primary analysis were also validated with Python 3.12.3,
NumPy 1.26.4, and SciPy 1.13.1. Compatible dependency ranges are listed in
`requirements.txt`; exact recorded GPU-stage versions are in
`requirements-training.txt`. Numerical identity across untested hardware and
library combinations is not assumed.

## Complete pipeline

The downloader obtains the fixed CAMUS release and its official split from the
publisher. It verifies downloaded file sizes and records content hashes. Dataset
terms and attribution are retained in [NOTICE.md](../NOTICE.md).

```bash
python -m data.download_camus
python -m data.prepare
python -m data.validate --check-reuse
python -m geometry.reproduce_publisher
python -m segmentation.infer --device cuda
python -m analysis.run
python -m analysis.figures
```

Outputs are written to `data/prepared/` and `outputs/`. Existing geometry,
prediction, and evaluation outputs are protected from overwrite; use a fresh
`--output` directory when repeating a run. On a CPU-only system, install the CPU
build of PyTorch and pass `--device cpu` to inference.

The geometry run should reproduce all 50 reference EF labels. The integrated
evaluation reports Dice, physical boundary distances, volume errors, EF agreement,
coverage, and patient-level confidence intervals. Reference measurements are
used only for evaluation; inference loads grayscale images without target arrays.

## Training and verification

Retraining uses the fixed training and validation partitions:

```bash
python -m segmentation.training --output outputs/retrained --device cuda
python -m segmentation.infer --run outputs/retrained --output outputs/retrained_predictions --device cuda
python -m pytest -q
python -m analysis.verify
```

The included checkpoint was selected by validation Dice before test evaluation.
Its original training metadata remains unchanged. A separate packaging record
documents bitwise equivalence of the released model on all 200 native test
frames. Content-based data identity permits regeneration of the same source,
split, and preprocessing configuration without requiring matching timestamps or
platform newline conventions.

`analysis.verify` checks the release manifest, checkpoint and notebook hashes,
saved geometry and analysis artifacts, figures, patient counts, and independently
recalculated summary statistics. It requires no raw images. To compare a new
inference run with the recorded prediction-file hashes, additionally pass
`--predictions outputs/predictions`; this strict check targets the recorded
numerical environment. The [validation record](../results/validation.json)
states the checks actually performed for this release.

Fresh preparation in the recorded GPU environment reproduced all 2,000 image,
mask, and valid-region arrays exactly. NPZ container bytes differed across NumPy
versions, while all 200 native prediction files and the patient-level results
remained unchanged.

## Controlled-error analysis

The fixed [protocol](perturbation_protocol.md) uses the same test patients and
published prediction files. No retraining is required. After data preparation
and inference, run the extension in a new output directory:

```bash
python -m analysis.sensitivity --output outputs/sensitivity_reproduction --workers 8
python -m analysis.sampling_audit --results outputs/sensitivity_reproduction
python -m analysis.sensitivity_figures --results outputs/sensitivity_reproduction --output outputs/sensitivity_reproduction/figures
python -m analysis.sensitivity_report --results outputs/sensitivity_reproduction
python -m analysis.verify_sensitivity --results outputs/sensitivity_reproduction --receipt outputs/sensitivity_reproduction/validation.json
```

Use `--prepared` and `--predictions` to select existing input directories. The
runner verifies the source hashes, fixed split, selected checkpoint, prediction
manifest, and native arrays before reporting results. It evaluates 55 scenarios
per patient with three methods and refuses to overwrite existing run outputs.
The sampling audit investigates the observed non-monotonic volume response of
the original estimator; it does not alter the main experiment.

The committed [report](../results/sensitivity_v1/REPORT.md) includes 8,250
attempts, 150 zero controls, 1,400 frame-variant audits, 588 prespecified
associations including strata, and seven PNG/SVG figures. The main cohort run
used Python 3.12.3, NumPy 1.26.4, SciPy 1.13.1, and scikit-image 0.23.2.
The publisher returned all 2,750 estimates. Two synthetic one-pixel holes caused
16 secondary-method rejections; those rows remain explicit and are never
replaced by zero. Means and intervals use available numeric patient pairs,
and paired method contrasts report their common coverage.

Both `python -m analysis.verify` and `python -m analysis.verify_sensitivity`
verify the saved extension without raw images using the Python standard library.
The latter independently recalculates paired differences, summaries, method
contrasts, rank correlations, and EF contribution identities. To also compare
against the original local archive, pass
`--archive ../evaluation/results/unet_v2_groupnorm_test` when that archive is
available. The portable package does not require it.

On Windows, a workspace-local pytest directory avoids sandbox temporary-file
rename restrictions:

```bash
python -m pytest -q --basetemp .work/pytest
```
