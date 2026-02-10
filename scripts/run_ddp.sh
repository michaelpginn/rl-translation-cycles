#!/bin/bash
#SBATCH --gres=gpu:a100:3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc-ddp
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --qos=blanca-curc-gpu

module load uv
cd "$SLURM_SUBMIT_DIR"
uv sync
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn install skipped"

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=4

uv run torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    run.py "$1" "${@:2}"
