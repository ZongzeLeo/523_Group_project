#!/bin/bash
#SBATCH -J camus_volumes
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
RUN=runs/unet_e80
# ------------------

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate test_env
set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python -c "import skimage" 2>/dev/null || pip install -q scikit-image

# 1) per-case table: demographics, reference EF, per-frame Dice
python make_table.py --data-root "$DATA_ROOT" --splits splits/splits.json \
    --metrics "$RUN/eval/metrics_per_frame.csv" --out "$RUN/eval/per_case_results.csv"

# 2) Simpson's biplane EDV / ESV / EF from predicted and expert masks, merged into the table
python compute_volume_simpson.py --data-root "$DATA_ROOT" --splits splits/splits.json \
    --pred-dir "$RUN/predictions" --out "$RUN/eval/volumes_per_case.csv" \
    --case-table