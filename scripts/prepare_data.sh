#!/bin/bash
# Run this on the login node to pre-download all datasets and model weights.
# After this, SLURM jobs can run with HF_DATASETS_OFFLINE=1.

set -e

cd "$(dirname "$0")/.."
module load uv 2>/dev/null || true
uv sync

echo "=== Downloading and caching FineWeb training sentences ==="
uv run python -c "
from src.config import ExperimentConfig
from src.data.loading import FineWebTrainDataset
config = ExperimentConfig()
FineWebTrainDataset(config)
print('Done.')
"

echo "=== Downloading FLORES-200 eval data ==="
uv run python -c "
from datasets import load_dataset
langs = {
    'yo': 'yor_Latn', 'ig': 'ibo_Latn', 'ha': 'hau_Latn',
    'zu': 'zul_Latn', 'xh': 'xho_Latn', 'gn': 'grn_Latn',
    'qu': 'que_Latn', 'ay': 'aym_Latn', 'lo': 'lao_Laoo',
    'my': 'mya_Mymr', 'km': 'khm_Khmr', 'am': 'amh_Ethi',
}
for code, flores_code in langs.items():
    print(f'Downloading eng_Latn-{flores_code}...')
    ds = load_dataset('facebook/flores', f'eng_Latn-{flores_code}', split='devtest', trust_remote_code=True)
    print(f'  {len(ds)} examples cached')
print('Done.')
"

echo "=== Downloading Qwen3-0.6B model ==="
uv run python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
print('Done.')
"

echo "=== All data downloaded and cached ==="
