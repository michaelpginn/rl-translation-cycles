#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu

module load uv
cd "$SLURM_SUBMIT_DIR"
uv sync
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install skipped"
uv run python run.py "$1" "${@:2}"
