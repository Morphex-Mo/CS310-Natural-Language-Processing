#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH -J A3_compare_offline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=titan

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export HUGGINGFACE_HUB_CACHE="${SLURM_SUBMIT_DIR}/.hf_cache"
export HF_HOME="${SLURM_SUBMIT_DIR}/.hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

[[ -f "wikizh_tokenizer_whitespace.json" ]] || { echo "Missing tokenizer file"; exit 1; }
[[ -d "$HUGGINGFACE_HUB_CACHE" ]] || { echo "Missing cache dir: $HUGGINGFACE_HUB_CACHE"; exit 1; }

PYTHONUNBUFFERED=1 python -u compare_tokenizers.py | tee compare_output_offline_job.txt
