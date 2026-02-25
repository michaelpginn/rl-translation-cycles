#!/bin/bash
#SBATCH --gres=gpu:h100_3g.40gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000m
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=rltc
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --qos=blanca-blast-lecs

module load uv
cd "$SLURM_SUBMIT_DIR"
uv sync
uv pip install flash-attn setuptools --no-build-isolation

echo "=== CUDA + PyTorch diagnostics ==="
uv run python - <<'PY'
import torch, os
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with:", torch.__config__.show())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Detected GPUs:", torch.cuda.device_count())
    print("GPU 0:", torch.cuda.get_device_name(0))
PY

uv run python run.py "$1" "${@:2}"
