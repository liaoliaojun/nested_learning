#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_MODEL=${1:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
TOKENIZER_DIR="$(dirname "${TOKENIZER_MODEL}")"
RPJ_DATASET=${RPJ_DATASET:-gmongaras/SlimPajama-627B_Reupload}

if [[ ! -f "data/filtered/refinedweb_en_sample.txt" ]]; then
  echo "[Data] Creating filtered RefinedWeb sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset HuggingFaceFW/fineweb \
    "--subset=sample-10BT" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 2000 \
    --output-path data/filtered/refinedweb_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/wikipedia_en_sample.txt" ]]; then
  echo "[Data] Creating filtered Wikipedia sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset wikimedia/wikipedia \
    "--subset=20231101.en" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/wikipedia_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/c4_en_sample.txt" ]]; then
  echo "[Data] Creating filtered C4 sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset allenai/c4 --subset en --split train \
    --text-column text --target-lang en --lang-threshold 0.85 \
    --min-chars 200 --max-chars 8000 --limit 1000 \
    --output-path data/filtered/c4_en_sample.txt --force-exit
fi

if [[ ! -f "data/filtered/redpajama_en_sample.txt" ]]; then
  echo "[Data] Creating filtered SlimPajama sample"
  # 兼容不同镜像配置：先尝试无 subset，再尝试 subset=default。
  if ! uv run python scripts/data/filter_corpus.py \
    "--dataset=${RPJ_DATASET}" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/redpajama_en_sample.txt \
    --force-exit; then
    echo "[Data][Warn] SlimPajama without subset failed; retrying with subset=default"
    if ! uv run python scripts/data/filter_corpus.py \
      "--dataset=${RPJ_DATASET}" \
      --subset default \
      --split train \
      --text-column text \
      --target-lang en \
      --lang-threshold 0.85 \
      --min-chars 200 \
      --max-chars 8000 \
      --limit 1000 \
      --output-path data/filtered/redpajama_en_sample.txt \
      --force-exit; then
    echo "[Data][Warn] SlimPajama unavailable; falling back to c4/wikipedia sample"
    if [[ -f "data/filtered/c4_en_sample.txt" ]]; then
      cp data/filtered/c4_en_sample.txt data/filtered/redpajama_en_sample.txt
    elif [[ -f "data/filtered/wikipedia_en_sample.txt" ]]; then
      cp data/filtered/wikipedia_en_sample.txt data/filtered/redpajama_en_sample.txt
    else
      echo "[Data][Error] No fallback corpus found for redpajama sample"
      exit 1
    fi
    fi
  fi
fi

if [[ ! -f "data/filtered/code_en_sample.txt" ]]; then
  echo "[Data] Creating filtered code sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset codeparrot/codeparrot-clean-train --split train \
    --text-column content --target-lang en --lang-threshold 0.5 \
    --min-chars 200 --max-chars 12000 --limit 1000 \
    --output-path data/filtered/code_en_sample.txt --force-exit
fi

if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
  echo "[Data] Training tokenizer (sample) -> ${TOKENIZER_DIR}"
  uv run python scripts/data/train_tokenizer.py \
    --manifest configs/data/refinedweb_mixture_filtered.yaml \
    --vocab-size 32000 \
    --no-hard-vocab-limit \
    --output-dir "${TOKENIZER_DIR}" \
    --log-file data/mixtures/refinedweb_mix_tokenizer_sample.json
fi

echo "[Data] Sharding filtered samples"
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path ${TOKENIZER_MODEL} \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json

echo "[Data] Sample pipeline complete"
