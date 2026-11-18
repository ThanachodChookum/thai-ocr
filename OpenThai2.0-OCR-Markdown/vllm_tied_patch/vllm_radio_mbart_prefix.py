# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""vLLM 0.26 C-RADIO + same-weight mBART causal visual-prefix adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os

import torch
import torch.nn as nn
from transformers import BatchFeature, PretrainedConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.nemotron_parse import (
    BartScaledWordEmbedding,
    RadioWithNeck,
    WhisperAttention,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.v1.attention.backend import AttentionType


from .mtp_weights import AUXILIARY_TENSOR_NAMES


IMAGE_TOKEN_ID = 1
IMAGE_TOKEN = "<pad>"


class MBartPrefixDecoderLayer(nn.Module):
    """The trained mBART layer with only its causal self-attention path."""

    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self.self_attn = WhisperAttention(
            embed_dim=config.d_model,
            num_heads=config.decoder_attention_heads,
            attn_type=AttentionType.DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        query_heads = int(config.decoder_attention_heads)
        total_kv_heads = int(
            getattr(config, "self_attention_kv_heads", query_heads)
        )
        tp_size = get_tensor_model_parallel_world_size()
        if query_heads % tp_size or query_heads % total_kv_heads:
            raise ValueError(
                "mBART prefix GQA requires TP and KV heads to divide query heads: "
                f"query_heads={query_heads}, total_kv_heads={total_kv_heads}, "
                f"tp_size={tp_size}"
            )
        if total_kv_heads < tp_size and tp_size % total_kv_heads:
            raise ValueError(
                f"TP size {tp_size} must be divisible by KV heads {total_kv_heads}"
            )
        expected_local_kv_heads = (
            total_kv_heads // tp_size
            if total_kv_heads >= tp_size
            else 1
        )
        actual_local_kv_heads = int(
            getattr(self.self_attn, "num_kv_heads", query_heads // tp_size)
        )
        if actual_local_kv_heads != expected_local_kv_heads:
            raise RuntimeError(
                "mBART prefix GQA mismatch: "
                f"query_heads={query_heads}, total_kv_heads={total_kv_heads}, "
                f"tp_size={tp_size}, expected_local_kv_heads="
                f"{expected_local_kv_heads}, actual_local_kv_heads="
                f"{actual_local_kv_heads}. Ensure the vLLM GQA patch runs "
                "before importing vllm_radio_mbart_prefix."
            )
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = get_act_fn(config.activation_function)
        self.fc1 = ColumnParallelLinear(
            config.d_model, config.decoder_ffn_dim, bias=True,
            quant_config=quant_config, prefix=f"{prefix}.fc1",
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim, config.d_model, bias=True,
            quant_config=quant_config, prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states=hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return residual + hidden_states


class MBartPrefixDecoder(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        scale = config.d_model**0.5 if config.scale_embedding else 1.0
        self.embed_tokens = BartScaledWordEmbedding(config.vocab_size, config.d_model, embed_scale=scale)
        self.layers = nn.ModuleList([
            MBartPrefixDecoderLayer(config, cache_config, quant_config, f"{prefix}.layers.{index}")
            for index in range(config.decoder_layers)
        ])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def embed_input_ids(self, input_ids):
        """Expose the causal decoder's embedding entrypoint to vLLM."""
        return self.embed_tokens(input_ids)

    @property
    def model(self):
        """Expose the inner-language-model contract used by vLLM MTP."""
        return self

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name.startswith("embed_positions") or ".encoder_attn" in name:
                continue
            for target, source, shard in ((".self_attn.qkv_proj", ".self_attn.q_proj", "q"), (".self_attn.qkv_proj", ".self_attn.k_proj", "k"), (".self_attn.qkv_proj", ".self_attn.v_proj", "v")):
                if source in name:
                    mapped = name.replace(source, target)
                    if mapped not in params:
                        raise KeyError(f"Unknown fused mBART prefix parameter: {mapped}")
                    params[mapped].weight_loader(params[mapped], weight, shard)
                    loaded.add(mapped)
                    break
            else:
                if name not in params:
                    raise KeyError(f"Unknown mBART prefix parameter: {name}")
                loader = getattr(params[name], "weight_loader", default_weight_loader)
                loader(params[name], weight)
                loaded.add(name)
        return loaded


def ensure_encoder_config(config):
    if isinstance(config.encoder, dict):
        config.encoder = PretrainedConfig.from_dict(config.encoder)
    if isinstance(config.decoder, dict):
        config.decoder = PretrainedConfig.from_dict(config.decoder)
    return config.encoder


class RadioMBartPrefixProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 1}

    def get_num_image_tokens(self, height: int = 2048, width: int = 1664) -> int:
        config = self.get_hf_config()
        encoder = ensure_encoder_config(config)
        compression = int(os.environ.get("NP_VLLM_NECK_COMPRESSION", getattr(config, "neck_compression", 4)))
        return (height // encoder.patch_size) * ((width // encoder.patch_size) // compression) + 1

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_num_image_tokens()}


class RadioMBartPrefixDummyInputsBuilder(BaseDummyInputsBuilder[RadioMBartPrefixProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_TOKEN * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions]) -> MultiModalDataDict:
        del seq_len
        return {"image": self._get_dummy_images(width=1664, height=2048, num_images=mm_counts.get("image", 0), overrides=mm_options.get("image"))}


class RadioMBartPrefixProcessor(BaseMultiModalProcessor[RadioMBartPrefixProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """The HF processor leaves visual-token expansion to vLLM."""
        return False

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs):
        image_sizes = hf_inputs.get("image_sizes", torch.empty((0, 2)))
        patch_counts = (image_sizes[:, 0] // 16) * (image_sizes[:, 1] // 16)
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes("image", patch_counts),
            "image_sizes": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
            "content_sizes": MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        }

    def _get_prompt_updates(self, mm_items: MultiModalDataItems, hf_processor_mm_kwargs: Mapping[str, object], out_mm_kwargs: MultiModalKwargsItems) -> Sequence[PromptUpdate]:
        del self, mm_items, hf_processor_mm_kwargs

        def replacement(item_idx: int):
            size = out_mm_kwargs["image"][item_idx]["image_sizes"].data
            height, width = (int(value) for value in size.reshape(-1).tolist())
            compression = int(os.environ.get("NP_VLLM_NECK_COMPRESSION", "4"))
            return [IMAGE_TOKEN_ID] * ((height // 16) * ((width // 16) // compression) + 1)

        return [PromptReplacement(modality="image", target=[IMAGE_TOKEN_ID], replacement=replacement)]


@MULTIMODAL_REGISTRY.register_processor(
    RadioMBartPrefixProcessor,
    info=RadioMBartPrefixProcessingInfo,
    dummy_inputs=RadioMBartPrefixDummyInputsBuilder,
)
class RadioMBartPrefixForCausalLM(nn.Module, SupportsMultiModal):
    """Use trained mBART causal blocks with visual embeddings in their KV cache."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_TOKEN
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        ensure_encoder_config(config)
        self.config = config
        self.vocab_size = int(config.decoder.vocab_size)
        self.configure_mm_token_handling(vocab_size=self.vocab_size, mm_token_ids=[IMAGE_TOKEN_ID])
        with self._mark_tower_model(vllm_config, "image"):
            self.encoder = RadioWithNeck(config=config, quant_config=vllm_config.quant_config, prefix=f"{prefix}.encoder")
        with self._mark_language_model(vllm_config):
            self.decoder = MBartPrefixDecoder(
                config.decoder,
                vllm_config.cache_config,
                vllm_config.quant_config,
                f"{prefix}.decoder",
            )
        self.lm_head = ParallelLMHead(
            self.vocab_size,
            config.decoder.d_model,
            quant_config=vllm_config.quant_config,
        )
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head = self.lm_head.tie_weights(self.decoder.embed_tokens)
            if self.lm_head.weight is not self.decoder.embed_tokens.weight:
                raise RuntimeError(
                    "vLLM failed to tie the mBART prefix LM head to decoder embeddings"
                )
            print(
                "[np-vllm-mbart-prefix] tied lm_head.weight to "
                "decoder.embed_tokens.weight",
                flush=True,
            )
        self.logits_processor = LogitsProcessor(self.vocab_size, self.vocab_size)

    def _parse_and_validate_image_input(self, **kwargs):
        pixels = kwargs.pop("pixel_values", None)
        sizes = kwargs.pop("image_sizes", None)
        content_sizes = kwargs.pop("content_sizes", None)
        if pixels is None:
            return None
        if sizes is None:
            raise ValueError("C-RADIO mBART prefix input requires image_sizes")
        if content_sizes is None:
            raise ValueError("C-RADIO mBART prefix input requires content_sizes")
        return {
            "data": pixels,
            "image_sizes": sizes,
            "content_sizes": content_sizes,
        }

    def get_language_model(self):
        """Return the decoder so vLLM can share its tied embedding/head with MTP."""
        return self.decoder

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        pixels = image_input["data"]
        if pixels.ndim == 2:
            pixels = pixels.unsqueeze(0)
        dtype = next(self.encoder.parameters()).dtype
        return self.encoder(
            pixels.to(dtype),
            image_sizes=image_input["image_sizes"].reshape(-1, 2).tolist(),
            content_sizes=image_input["content_sizes"].reshape(-1, 2).tolist(),
        )

    def _prefix_decoder(self, input_ids, inputs_embeds):
        if inputs_embeds is None:
            inputs_embeds = self.decoder.embed_tokens(input_ids)
        hidden_states = self.decoder.layernorm_embedding(inputs_embeds)
        for layer in self.decoder.layers:
            hidden_states = layer(hidden_states)
        return self.decoder.layer_norm(hidden_states)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        del positions, intermediate_tensors, kwargs
        return self._prefix_decoder(input_ids, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        encoder_weights, decoder_weights = [], []
        lm_head = dict(self.lm_head.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            # Auxiliary tensors belong to the MTP proposer, which reads the
            # same checkpoint. They are not parameters of the target model.
            if name in AUXILIARY_TENSOR_NAMES:
                continue
            if name.startswith("encoder."):
                encoder_weights.append((name[len("encoder."):], weight))
            elif name.startswith("decoder."):
                decoder_weights.append((name[len("decoder."):], weight))
            elif name.startswith("lm_head."):
                short = name[len("lm_head."):]
                if short not in lm_head:
                    raise KeyError(f"Unknown mBART prefix LM-head parameter: {short}")
                default_weight_loader(lm_head[short], weight)
                loaded.add(name)
            else:
                raise KeyError(f"Unexpected mBART prefix checkpoint tensor: {name}")

        # vLLM 0.26's RadioWithNeck loader does not return its delegated load
        # set, so reproduce its small routing layer here and retain RADIO's own
        # strict fused-parameter accounting.
        encoder_adaptor = {
            name: param
            for name, param in self.encoder.named_parameters()
            if not name.startswith("model_encoder")
        }
        radio_weights = []
        for name, weight in encoder_weights:
            if name.startswith("model_encoder."):
                radio_weights.append((name[len("model_encoder."):], weight))
                continue
            if name not in encoder_adaptor:
                raise KeyError(f"Unknown mBART prefix encoder parameter: {name}")
            loader = getattr(
                encoder_adaptor[name], "weight_loader", default_weight_loader
            )
            loader(encoder_adaptor[name], weight)
            loaded.add(f"encoder.{name}")
        radio_loaded = self.encoder.model_encoder.load_weights(radio_weights)
        # C-RADIOv4-SO400M's timm backbone is constructed without LayerScale:
        # every block uses nn.Identity and its checkpoint consequently contains
        # no ls1/ls2 tensors.  vLLM's shared InternViT layer always materializes
        # identity-initialized ls1/ls2 Parameters, while RadioModel.load_weights
        # deliberately skips LayerScale entries.  Account for those synthetic
        # parameters only after proving that this is the SO400M architecture and
        # that every value is exactly the source-equivalent identity value.
        encoder_args = getattr(self.config.encoder, "args", {})
        radio_model_name = (
            encoder_args.get("model")
            if isinstance(encoder_args, Mapping)
            else getattr(encoder_args, "model", None)
        )
        identity_layerscale_models = {
            "vit_so400m_patch16_224",
            "vit_so400m_v2_patch16_224",
        }
        radio_source_names = {name for name, _ in radio_weights}
        synthetic_layerscales = {
            name: param
            for name, param in self.encoder.model_encoder.named_parameters()
            if name.endswith((".ls1", ".ls2")) and name not in radio_loaded
        }
        if synthetic_layerscales:
            source_layerscales = {
                name
                for name in radio_source_names
                if ".ls1" in name or ".ls2" in name
            }
            if radio_model_name not in identity_layerscale_models:
                raise RuntimeError(
                    "vLLM exposed RADIO LayerScale parameters absent from its load "
                    f"set for unsupported architecture {radio_model_name!r}: "
                    f"{sorted(synthetic_layerscales)}"
                )
            if source_layerscales:
                raise RuntimeError(
                    "SO400M checkpoint unexpectedly contains LayerScale tensors that "
                    "vLLM ignored: "
                    f"{sorted(source_layerscales)}"
                )
            non_identity = [
                name
                for name, param in synthetic_layerscales.items()
                if not torch.equal(param.detach(), torch.ones_like(param.detach()))
            ]
            if non_identity:
                raise RuntimeError(
                    "vLLM synthetic SO400M LayerScale parameters are not identity: "
                    f"{non_identity}"
                )
            radio_loaded.update(synthetic_layerscales)
            print(
                "[np-vllm-mbart-prefix] accounted for "
                f"{len(synthetic_layerscales)} identity-only SO400M LayerScale "
                "parameters",
                flush=True,
            )
        loaded.update(f"encoder.model_encoder.{name}" for name in radio_loaded)

        decoder_loaded = self.decoder.load_weights(decoder_weights)
        loaded.update(f"decoder.{name}" for name in decoder_loaded)
        if bool(getattr(self.config, "tie_word_embeddings", False)):
            if self.lm_head.weight is not self.decoder.embed_tokens.weight:
                raise RuntimeError(
                    "mBART prefix LM-head tie was lost while loading checkpoint weights"
                )
        return loaded

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(language_model="decoder", tower_model="encoder")
