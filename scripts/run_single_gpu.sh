#!/bin/bash
#SBATCH --gres=gpu:h100_7g.80gb
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=4000m
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs

module load uv
cd "$SLURM_SUBMIT_DIR"
uv sync
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install skipped"
export CUDA_VISIBLE_DEVICES=1

uv run python run.py "$1" "${@:2}"
