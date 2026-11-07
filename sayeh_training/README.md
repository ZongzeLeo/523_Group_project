# CAMUS baseline U-Net: LV segmentation + Dice analysis

Baseline 2D U-Net trained on CAMUS ED/ES frames (2CH + 4CH, one model), with
evaluation stratified by view/phase, LVEF category and image quality.

## Setup
```bash
conda activate test_env
pip install torch nibabel scipy pandas matplotlib tabulate
```

## Run
Edit `DATA_ROOT` (the folder containing `patient0001/`, ...) in `train_and_eval.sbatch`, then:
```bash
sbatch train_and_eval.sbatch
```
Or step by step:
```bash
python make_splits.py   --data-root $D --out-dir splits --split-csv splits/team_splits.csv
python train.py         --data-root $D --splits splits/splits.json --out-dir runs/unet_baseline
python predict.py       --data-root $D --checkpoint runs/unet_baseline/best.pt --split val,test
python evaluate_dice.py --data-root $D --split test
```
`splits/team_splits.csv` is the team repo's `data/reference/splits.csv`: the official CAMUS split
(400 train / 50 val / 50 test patients = 1,600 / 200 / 200 frames), so results are on exactly the
same test patients as the rest of the project. Alternatives: `--official-split-dir <CAMUS>/database_split`,
or no flag for an EF × quality stratified split.

## Files
| Script | Does |
|---|---|
| `make_splits.py` | Patient-level split (official CAMUS lists; checks partitions are disjoint and every patient exists). Writes `splits.json` and `patients.csv` (EF, reference volumes, per-view quality, split) and prints EF category counts per split. |
| `dataset.py` | Intensities / 255, isotropic physical-aspect-preserving letterbox into 256×256 (images bilinear, masks nearest), valid field-of-view mask; augmentation. |
| `train.py` | U-Net (depth 5, 32 base channels, BN), CE + soft Dice restricted to the valid FOV, AdamW + warmup/cosine, bf16 autocast on GPU, early stopping on val LV Dice. |
| `predict.py` | Predicts, crops the letterbox and resamples **probabilities** back to the native grid (inverse transform), saves NIfTI masks with the **original header** (pixel spacing preserved). |
| `evaluate_dice.py` | Dice, Hausdorff, HD95 and MASD (mm) at native resolution; summary tables + figure. |

## For teammates (geometric / volume analysis)
- Predicted masks: `runs/unet_baseline/predictions/<pid>/<pid>_<view>_<phase>_pred.nii.gz`,
  same shape/affine/spacing as the CAMUS `_gt` files. Read spacing with `img.header.get_zooms()[:2]`.
- Labels match CAMUS GT: **1 = LV endocardium (use this for volumes)**, 2 = myocardium, 3 = LA.
- `_pred` = post-processed (largest LV component + holes filled); `_pred_raw` = plain argmax.
  `predictions_manifest.csv` records how many LV fragments each raw mask had.
- Use the shared `splits/splits.json` and `patients.csv` so everyone analyses the same test patients.

## For the write-up
`runs/unet_baseline/config.json` records everything the methods section needs:
architecture and parameter count, loss, optimizer and hyperparameters, LR schedule,
augmentation, preprocessing, split method and sizes, **hardware** (GPU name/memory,
CUDA/cuDNN, torch version, CPUs, SLURM job ID), best epoch and training time.
`eval/summary_lv.md` has ready-made tables; `eval/lv_dice_hd_boxplots.png` is the figure.

Notes:
- `n` in the summary tables counts **frames** (4 per patient), not patients.
- Resampling alone limits Dice: letterboxing a GT mask to 256×256 and mapping it back gives
  LV Dice ≈ 0.99, not 1.0. Worth one sentence in the write-up as the ceiling from preprocessing.
- Validation Dice during training is computed at the training resolution (256×256) and
  is only used for model selection. The reported numbers come from `evaluate_dice.py` at native resolution.
