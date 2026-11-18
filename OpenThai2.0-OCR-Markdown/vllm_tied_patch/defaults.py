# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Model-scoped serving defaults for Nemotron 3.5 Parse."""

from __future__ import annotations

import os


def is_parse_config(config) -> bool:
    return bool(
        {"RadioMBartPrefixForCausalLM", "ParseAuxMTPModel"}
        & set(getattr(config, "architectures", []) or [])
    )


def disable_fa3_aot(model_config) -> bool:
    """Use the validated FA3 schedule for this model and its MTP proposer."""
    return is_parse_config(getattr(model_config, "hf_config", None)) and (
        os.environ.get("NP_VLLM_FA3_DISABLE_AOT_SCHEDULER", "1") == "1"
    )


def default_mtp(engine_args, model_config) -> bool:
    """Supply one-token MTP only when no speculative options were specified."""
    config = getattr(model_config, "hf_config", None)
    return (
        is_parse_config(config)
        and int(getattr(config, "num_extra_heads", 0) or 0) > 0
        and os.environ.get("NP_VLLM_MTP", "1") == "1"
        and engine_args.speculative_config is None
        and all(getattr(engine_args, name, None) is None
                for name in ("spec_method", "spec_model", "spec_tokens"))
    )


def register_serving_defaults() -> None:
    from vllm.engine.arg_utils import EngineArgs

    original = EngineArgs.create_speculative_config
    if getattr(original, "_np_parse_default_mtp", False):
        return

    def create_speculative_config(self, target_model_config, target_parallel_config):
        if default_mtp(self, target_model_config):
            self.speculative_config = {"method": "mtp", "num_speculative_tokens": 1}
            print("[nemotron-parse] one-token MTP enabled by default", flush=True)
        return original(self, target_model_config, target_parallel_config)

    create_speculative_config._np_parse_default_mtp = True
    EngineArgs.create_speculative_config = create_speculative_config
