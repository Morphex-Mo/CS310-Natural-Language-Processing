#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH -J A3_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=titan

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Optional: module load python/3.x
python preprocess_wikizh.py \
  --input "/home/xuyang_lab/cse12212752/A3-code/CS310-Natural-Language-Processing/coding assignment/A3/wiki_zh_2019/wiki_zh" \
  --output "wikizh.txt"

wc -l wikizh.txt
wc -w wikizh.txt
