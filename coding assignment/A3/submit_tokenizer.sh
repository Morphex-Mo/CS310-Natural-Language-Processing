#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH -J A3_tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=titan

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Use local cache first so compare_tokenizers can run without external network.
export HUGGINGFACE_HUB_CACHE="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# Optional: module load python/3.x
PYTHONUNBUFFERED=1 python -u -c "import tokenizers, transformers; print('tokenizer_env_ok')"

if [[ -f "wikizh_tokenizer_whitespace.json" ]]; then
  echo "Tokenizer already exists: wikizh_tokenizer_whitespace.json (skip retraining)"
else
  PYTHONUNBUFFERED=1 python -u train_tokenizer_from_scratch.py \
    --input "wikizh.txt" \
    --vocab_size 52000 \
    --pre_tokenizer Whitespace \
    --output "wikizh_tokenizer_whitespace.json"
fi

echo "Running compare_tokenizers in offline mode with cache: $HUGGINGFACE_HUB_CACHE"
PYTHONUNBUFFERED=1 python -u compare_tokenizers.py

PYTHONUNBUFFERED=1 python -u count_tokens_stream.py \
  --tokenizer "wikizh_tokenizer_whitespace.json" \
  --input "wikizh.txt" \
  --output_json "token_stats_full.json"
