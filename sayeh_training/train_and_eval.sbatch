#!/bin/bash
#SBATCH -J camus_unet
#SBATCH -p mig
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_b200_3g.90gb:1
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%j_out.txt
#SBATCH -e logs/%j_err.txt

set -euo pipefail

# --- edit these ---
DATA_ROOT=/data/scratch/users/sayeh/CAMUS/database_nifti
SPLIT_CSV=splits/team_splits.csv   # team repo's data/reference/splits.csv (official 400/50/50)
RUN=runs/unet_e80
# ------------------

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test_env
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

nvidia-smi

# 1) patient-level split (skipped if it already exists so teammates share one split)
if [ ! -f splits/splits.json ]; then
    python make_splits.py --data-root "$DATA_ROOT" --out-dir splits --split-csv "$SPLIT_CSV"
fi

# 2) train
#python train.py --data-root "$DATA_ROOT" --splits splits/splits.json --out-dir "$RUN" \
#    --num-workers "${SLURM_CPUS_PER_TASK:-8}"
python train.py --data-root "$DATA_ROOT" --splits splits/splits.json --out-dir "$RUN" \
     --num-workers "${SLURM_CPUS_PER_TASK:-8}" --epochs 80 --patience 80

# 3) predict test (and val, useful for the error-propagation analysis)
python predict.py --data-root "$DATA_ROOT" --splits splits/splits.json \
    --checkpoint "$RUN/best.pt" --out-dir "$RUN/predictions" --split val,test

# 4) Dice / HD / MASD on the test set
python evaluate_dice.py --data-root "$DATA_ROOT" --splits splits/splits.json \
    --pred-dir "$RUN/predictions" --out-dir "$RUN/eval" --split test
