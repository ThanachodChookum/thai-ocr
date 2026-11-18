# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""One-step Nemotron 3.5 Parse auxiliary-head proposer for vLLM MTP."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn
from .mtp_weights import load_auxiliary_weights

from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


class _ParseAuxMTPInner(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.extra_proj = nn.Linear(hidden_size, hidden_size)
        self.extra_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        next_token_embeds = self.embed_tokens(input_ids)
        return self.extra_head(hidden_states + self.extra_proj(next_token_embeds))


class ParseAuxMTPModel(nn.Module):
    """Draft model backed by Nemotron 3.5 Parse's trained one-step auxiliary head."""

    # vLLM replaces both temporary modules with the target model's tied
    # decoder embedding and LM head after loading, so there is no persistent
    # duplicate vocabulary matrix at inference time.
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        config = vllm_config.model_config.hf_config
        decoder = getattr(config, "decoder", None)
        vocab_size = int(
            getattr(decoder, "vocab_size", getattr(config, "vocab_size"))
        )
        hidden_size = int(
            getattr(decoder, "d_model", getattr(config, "hidden_size"))
        )
        self.model = _ParseAuxMTPInner(vocab_size, hidden_size)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(vocab_size, vocab_size)
        self.model_path = Path(vllm_config.model_config.model)
        self.model_revision = getattr(vllm_config.model_config, "revision", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del positions, inputs_embeds, kwargs
        return self.model(input_ids, hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        mapping = {
            "decoder.extra_heads.0.weight": self.model.extra_head.weight,
            "decoder.extra_heads.0.bias": self.model.extra_head.bias,
            "decoder.extra_proj.0.weight": self.model.extra_proj.weight,
            "decoder.extra_proj.0.bias": self.model.extra_proj.bias,
        }
        return load_auxiliary_weights(
            weights,
            embedding=self.model.embed_tokens.weight,
            parameters=mapping,
            weight_loader=default_weight_loader,
            model_path=self.model_path,
            revision=self.model_revision,
        )
