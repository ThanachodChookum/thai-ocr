# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Strict loading of the trained one-step MTP tensors.

New releases store these tensors in the ordinary model safetensors. The
sidecar fallback supports older checkpoints; partial integrated heads fail
instead of combining tensors from different checkpoint representations.
"""

from pathlib import Path

from safetensors import safe_open


AUXILIARY_TENSOR_NAMES = frozenset(
    {
        "decoder.extra_heads.0.weight",
        "decoder.extra_heads.0.bias",
        "decoder.extra_proj.0.weight",
        "decoder.extra_proj.0.bias",
    }
)


def load_auxiliary_weights(
    weights, *, embedding, parameters, weight_loader, model_path, revision=None
):
    if set(parameters) != AUXILIARY_TENSOR_NAMES:
        raise ValueError("MTP parameter mapping must contain the exact four auxiliary tensors")
    loaded = set()
    auxiliary_found = set()

    def load_auxiliary(name, tensor):
        if name in auxiliary_found:
            raise RuntimeError(f"Duplicate integrated MTP tensor: {name}")
        parameter = parameters[name]
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise ValueError(
                f"MTP tensor shape mismatch for {name}: "
                f"checkpoint={tuple(tensor.shape)} model={tuple(parameter.shape)}"
            )
        weight_loader(parameter, tensor)
        auxiliary_found.add(name)
        suffix = name.rsplit(".", 1)[-1]
        module = "extra_head" if ".extra_heads." in name else "extra_proj"
        loaded.add(f"model.{module}.{suffix}")

    for name, tensor in weights:
        if name == "decoder.embed_tokens.weight":
            weight_loader(embedding, tensor)
            loaded.add("model.embed_tokens.weight")
        elif name in parameters:
            load_auxiliary(name, tensor)
        elif name.startswith(("decoder.extra_heads.", "decoder.extra_proj.")):
            raise RuntimeError(f"Unsupported MTP tensor: {name}; this proposer supports one head")

    if auxiliary_found and auxiliary_found != AUXILIARY_TENSOR_NAMES:
        raise RuntimeError(
            f"Incomplete integrated MTP head; missing {sorted(AUXILIARY_TENSOR_NAMES - auxiliary_found)}"
        )
    if not auxiliary_found:
        sidecar_name = "auxiliary_prediction_heads.safetensors.extra"
        sidecar = Path(model_path) / sidecar_name
        if not sidecar.is_file() and not Path(model_path).exists():
            from huggingface_hub import hf_hub_download

            sidecar = Path(hf_hub_download(str(model_path), filename=sidecar_name, revision=revision))
        if not sidecar.is_file():
            raise FileNotFoundError(
                "Checkpoint has no integrated MTP tensors and no legacy auxiliary sidecar: "
                f"{sidecar}"
            )
        with safe_open(str(sidecar), framework="pt", device="cpu") as source:
            if set(source.keys()) != AUXILIARY_TENSOR_NAMES:
                raise RuntimeError(f"Unexpected legacy MTP sidecar tensors: {sorted(source.keys())}")
            for name in sorted(AUXILIARY_TENSOR_NAMES):
                load_auxiliary(name, source.get_tensor(name))
    if "model.embed_tokens.weight" not in loaded:
        raise RuntimeError("MTP checkpoint is missing decoder.embed_tokens.weight")
    return loaded
