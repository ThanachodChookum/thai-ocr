# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

FROM vllm/vllm-openai:v0.26.0
LABEL org.opencontainers.image.title="Nemotron 3.5 Parse" \
      org.opencontainers.image.version="3.5.0.post7" \
      org.opencontainers.image.licenses="OpenMDW-1.1"
WORKDIR /workspace
RUN python3 -m venv /opt/nemotron-parse-venv
ENV PATH="/opt/nemotron-parse-venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/nemotron-parse-venv" \
    PYTHONPATH=""
COPY pyproject.toml LICENSE THIRD_PARTY_NOTICES.md TOKENIZER_ATTRIBUTION.md postprocessing.py latex2html.py /tmp/nemotron-package/
COPY nemotron_parse /tmp/nemotron-package/nemotron_parse
COPY vllm_tied_patch /tmp/nemotron-package/vllm_tied_patch
RUN python3 -m pip install "/tmp/nemotron-package[vllm,client]" && python3 -m pip check
COPY . /workspace
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "/workspace", "--served-model-name", "nemotron-parse", "--trust-remote-code", "--dtype", "bfloat16", "--distributed-executor-backend", "uni", "--tensor-parallel-size", "1", "--max-model-len", "16384", "--max-num-seqs", "192", "--max-num-batched-tokens", "16384", "--no-enable-prefix-caching", "--attention-config", "{\"flash_attn_version\":3}", "--limit-mm-per-prompt", "{\"image\":1}", "--chat-template", "/workspace/chat_template.jinja"]
