#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc-eval

cd "$SLURM_SUBMIT_DIR"
uv run python run.py "$1" "${@:2}"
