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
