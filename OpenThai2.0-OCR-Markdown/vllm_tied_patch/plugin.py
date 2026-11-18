# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""vLLM general-plugin entry point for Nemotron 3.5 Parse.

Installing this repository registers the target visual-prefix model and its
one-step MTP proposer in every vLLM process. This replaces the implicit
``sitecustomize``/``PYTHONPATH`` startup hook with vLLM's documented plugin
lifecycle while preserving the validated vLLM 0.26 compatibility patches.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet


_SUPPORTED_RUNTIME = {
    "vllm": ">=0.25.0,<0.33",
    "transformers": ">=5.14.1,<5.20",
    "huggingface-hub": ">=1.24.0,<2",
}


def _validate_runtime_versions() -> None:
    """Reject unsupported serving versions before applying internal hooks."""
    mismatches = []
    for package, expected in _SUPPORTED_RUNTIME.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not installed"
        if actual == "not installed" or not SpecifierSet(expected).contains(actual, prereleases=False):
            mismatches.append(f"{package}=={actual} (requires {expected})")
    if mismatches:
        raise RuntimeError(
            "Nemotron 3.5 Parse vLLM plugin requires a supported runtime: "
            + ", ".join(f"{name}{value}" for name, value in _SUPPORTED_RUNTIME.items())
            + ". Found: "
            + ", ".join(mismatches)
        )


def _register_mtp() -> None:
    from vllm.config.speculative import SpeculativeConfig
    from vllm.model_executor.models import ModelRegistry

    if "ParseAuxMTPModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "ParseAuxMTPModel",
            "vllm_tied_patch.vllm_radio_mbart_mtp:ParseAuxMTPModel",
        )

    original_override = SpeculativeConfig.hf_config_override
    if getattr(original_override, "_np_parse_aux_mtp", False):
        return

    def parse_mtp_override(hf_config):
        architectures = getattr(hf_config, "architectures", []) or []
        if "RadioMBartPrefixForCausalLM" in architectures:
            hf_config.model_type = "mtp"
            hf_config.update(
                {
                    "architectures": ["ParseAuxMTPModel"],
                    "n_predict": 1,
                    "num_nextn_predict_layers": 1,
                }
            )
            return hf_config
        return original_override(hf_config)

    parse_mtp_override._np_parse_aux_mtp = True
    SpeculativeConfig.hf_config_override = staticmethod(parse_mtp_override)


def register() -> None:
    """Register 3.5 Parse target and draft models in the current process."""
    _validate_runtime_versions()
    # Importing the compatibility module applies the pinned vLLM 0.26 SO400M,
    # NaViT, GQA, target-model registration, and optional FA3 compatibility
    # hooks. Unlike top-level sitecustomize discovery, this is invoked by
    # vLLM's explicit general-plugin loader in every process.
    from . import compatibility as _compat  # noqa: F401

    _register_mtp()
    from .defaults import register_serving_defaults

    register_serving_defaults()
