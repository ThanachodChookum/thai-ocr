#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
vllm serve ./parse35 \
  --served-model-name nemotron-parse \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.8 \
  --enable-prefix-caching \
  --mm-processor-cache-gb 4 \
  --enable-chunked-prefill \
  --async-scheduling \
  --attention-config '{"backend": "FLASH_ATTN", "flash_attn_version": 3}' \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 17, 32, 34, 64, 68, 128, 136, 256, 272, 512, 544, 1024, 1088, 2048, 2176]}' \
  --speculative-config '{"method": "dflash", "model": "./parse35/dflash2", "num_speculative_tokens": 16}'
