#!/usr/bin/env bash
# Start Nemotron 3.5 Transcribe behind vLLM with CTC word timestamps enabled.
# Run on the GPU host, inside the NeMo Speech environment from the model card's
# "vLLM" setup (vLLM 0.32.0 + NeMo Speech main). Settings target one H100 80GB;
# add `--data-parallel-size 8 --api-server-count 8 --enable-expert-parallel
# --disable-custom-all-reduce` for 8 GPUs.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-./NVIDIA-Nemotron-3.5-Transcribe-30B-A4B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
# Word timestamps require the V1 model runner.
export VLLM_USE_V2_MODEL_RUNNER=0

# MTP2 rather than the card's MTP8 default: it has higher batch throughput on H100
# (RTFx ~410 vs ~270), which is what dataset generation needs.

exec vllm serve "$MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.94 \
  --max-model-len 65536 --max-num-batched-tokens 4096 --max-num-seqs 192 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none \
  --mm-processor-cache-gb 0 --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware \
  "$@"
