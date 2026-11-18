# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Runtime patch for tied Nemotron Parse lm_head in vLLM.

Current vLLM Nemotron Parse builds create a standalone ParallelLMHead and only
load it when `lm_head.weight` is materialized. Compact Nemotron Parse exports
tie the output head to `decoder.embed_tokens.weight` instead, so this patch
restores that tie after vLLM loads decoder weights.
"""

from __future__ import annotations


def _patch_nemotron_parse_tied_lm_head() -> None:
    try:
        import vllm.model_executor.models.nemotron_parse as np
    except Exception:
        return

    cls = getattr(np, "NemotronParseForConditionalGeneration", None)
    if cls is None:
        return

    original = getattr(cls, "load_weights", None)
    if original is None or getattr(original, "_np_tied_lm_head_patch", False):
        return

    def load_weights_with_tied_lm_head(self, weights):
        buffered_weights = list(weights)
        saw_lm_head = any(name.startswith("lm_head") for name, _ in buffered_weights)
        result = original(self, buffered_weights)

        decoder_cfg = getattr(getattr(self, "config", None), "decoder", None)
        tie_word_embeddings = bool(
            getattr(getattr(self, "config", None), "tie_word_embeddings", False)
            or getattr(decoder_cfg, "tie_word_embeddings", False)
        )
        if tie_word_embeddings and not saw_lm_head:
            self.lm_head.weight = self.decoder.embed_tokens.weight
            print(
                "[np-vllm-tie-patch] tied lm_head.weight to decoder.embed_tokens.weight",
                flush=True,
            )
        return result

    load_weights_with_tied_lm_head._np_tied_lm_head_patch = True
    cls.load_weights = load_weights_with_tied_lm_head


_patch_nemotron_parse_tied_lm_head()


"""Make the vLLM Nemotron Parse neck honor persisted C-RADIO dimensions."""


def _register_so400m_radio_config() -> None:
    """Register the C-RADIOv4-SO400M timm dimensions for vLLM 0.26.

    vLLM constructs ``RadioConfig`` before it constructs the patched Parse
    neck.  Registering the model here is therefore required for every
    SO400M export, including ordinary global-attention NaViT inference.  The
    same registration used to live only in the optional ViTDet study patch,
    which left production global-attention exports unable to start.
    """
    try:
        import vllm.transformers_utils.configs.radio as radio_config
    except Exception:
        return

    model_name = "vit_so400m_patch16_224"
    dimensions = (1152, 27, 16, 4304)
    existing = radio_config.VIT_TIMM_DIM_BY_NAME.get(model_name)
    if existing is not None and tuple(existing) != dimensions:
        raise RuntimeError(
            f"unexpected vLLM RADIO dimensions for {model_name}: {existing}"
        )
    radio_config.VIT_TIMM_DIM_BY_NAME[model_name] = dimensions
    print(
        "[np-vllm-radio-config] registered "
        f"{model_name} hidden=1152 layers=27 heads=16 intermediate=4304",
        flush=True,
    )


def _patch_np_architecture_neck_init():
    try:
        import torch.nn as nn
        import vllm.model_executor.models.nemotron_parse as np
    except Exception:
        return
    cls = getattr(np, "RadioWithNeck", None)
    if cls is None or getattr(cls, "_np_arch_init_patch", False):
        return
    original_init = cls.__init__

    def architecture_init(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        metadata = getattr(config, "architecture_benchmark", {}) or {}
        spatial_dim = int(metadata.get("spatial_dim") or self.conv1.in_channels)
        summary_dim = int(metadata.get("summary_dim") or config.encoder.summary_dim)
        if spatial_dim != self.conv1.in_channels:
            self.conv1 = nn.Conv1d(
                spatial_dim,
                self.conv1.out_channels,
                1,
                bias=self.conv1.bias is not None,
            )
        if summary_dim != self.sum_proj.weight.shape[1]:
            self.sum_proj = nn.Linear(
                summary_dim,
                self.sum_proj.weight.shape[0],
                bias=getattr(self.sum_proj, "bias", None) is not None,
            )

    cls.__init__ = architecture_init
    cls._np_arch_init_patch = True
    print("[np-vllm-arch] configurable C-RADIO neck enabled", flush=True)


_register_so400m_radio_config()
_patch_np_architecture_neck_init()


"""Runtime NaViT adapter for the vLLM 0.26 Nemotron Parse model.

The underlying vLLM RADIO implementation already supports packed dynamic
rectangles through ``imgs_sizes`` and variable-length vision attention.  The
Nemotron Parse adapter does not expose that path, so this module connects its
processor metadata, multimodal field layout, RADIO neck, and prompt lengths.
This compatibility module is loaded explicitly by the installed vLLM plugin.
"""


import os


# Replaced with the export's persisted geometry contract during packaging. Keeping it explicit makes old content-normalized exports possible.
_NP_NAVIT_COORDINATE_FRAME = "content_normalized"
_NP_NAVIT_POSITION_EMBEDDING_MODE = "content_normalized_2d_interpolate"
_NP_NECK_COMPRESSION = int(os.environ.get("NP_VLLM_NECK_COMPRESSION", "4"))
_NP_BACKBONE = os.environ.get("NP_VLLM_BACKBONE", "nvidia/C-RADIOv4-SO400M")
_NP_ATTENTION_MODE = os.environ.get("NP_VLLM_VISION_ATTENTION", "global")
_NP_VITDET_WINDOW = int(os.environ.get("NP_VLLM_VITDET_WINDOW", "0"))
_NP_VITDET_GLOBAL_LAYERS = int(os.environ.get("NP_VLLM_VITDET_GLOBAL_LAYERS", "4"))
_NP_MASKED_PARTIAL_POOL = os.environ.get("NP_VLLM_MASKED_PARTIAL_POOL", "0") == "1"


def _normalized_2d_position_pages(
    torch_module,
    *,
    patches,
    pos_embed,
    source_rows: int,
    source_cols: int,
    image_sizes,
    patch_size: int,
    content_sizes=None,
    content_normalized: bool = False,
):
    """Mirror training's per-page normalized-2D position interpolation.

    Keep this helper independent of vLLM imports so the export contract can be
    exercised on the login node.  vLLM supplies one concatenated raw-patch
    sequence, whereas training embeds each page separately; splitting before
    interpolation makes the two representations numerically identical.
    """

    source_rows = int(source_rows)
    source_cols = int(source_cols)
    patch_size = int(patch_size)
    if source_rows <= 0 or source_cols <= 0 or patch_size <= 0:
        raise ValueError(
            "normalized 2D positions require positive source grid and patch size"
        )
    if source_rows * source_cols != int(pos_embed.shape[1]):
        raise ValueError(
            "RADIO learned positional field shape is inconsistent: "
            f"source={source_rows}x{source_cols} tokens={int(pos_embed.shape[1])}"
        )

    positional_grid = pos_embed.reshape(
        1, source_rows, source_cols, -1
    ).permute(0, 3, 1, 2)
    normalized_content_sizes = None
    if content_sizes is not None:
        normalized_content_sizes = [
            tuple(int(value) for value in size) for size in content_sizes
        ]
        if len(normalized_content_sizes) != len(image_sizes):
            raise ValueError(
                "Content-size count must match image-size count: "
                f"content_sizes={len(normalized_content_sizes)} "
                f"image_sizes={len(image_sizes)}"
            )
    if content_normalized and normalized_content_sizes is None:
        raise ValueError("content-normalized 2D positions require content_sizes")

    positioned_pages = []
    cursor = 0
    for image_index, (raw_height, raw_width) in enumerate(image_sizes):
        height, width = int(raw_height), int(raw_width)
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"Page shape {(height, width)} is not aligned to patch size {patch_size}"
            )
        rows, cols = height // patch_size, width // patch_size
        patch_count = rows * cols
        page = patches[:, cursor : cursor + patch_count]
        if int(page.shape[1]) != patch_count:
            raise ValueError(
                "Normalized position split ended inside a page: "
                f"page={(height, width)} expected={patch_count} actual={int(page.shape[1])}"
            )
        cursor += patch_count
        if content_normalized:
            content_height, content_width = normalized_content_sizes[image_index]
            if content_height <= 0 or content_width <= 0:
                raise ValueError(
                    "content size must be positive, got "
                    f"{(content_height, content_width)}"
                )
            if content_height > height or content_width > width:
                raise ValueError(
                    "content size cannot exceed padded image: "
                    f"content={(content_height, content_width)} image={(height, width)}"
                )
            content_rows = (content_height + patch_size - 1) // patch_size
            content_cols = (content_width + patch_size - 1) // patch_size
            if min(content_rows, content_cols) == 1:
                # Match training's safe area-preserving path for augmentation
                # crops whose visible content spans only one patch in an axis.
                page_positions = torch_module.nn.functional.adaptive_avg_pool2d(
                    positional_grid.float(), (content_rows, content_cols)
                )
            else:
                page_positions = torch_module.nn.functional.interpolate(
                    positional_grid.float(),
                    size=(content_rows, content_cols),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            row_starts = (
                torch_module.arange(rows, device=page.device, dtype=torch_module.float32)
                * patch_size
            )
            col_starts = (
                torch_module.arange(cols, device=page.device, dtype=torch_module.float32)
                * patch_size
            )
            row_extents = (content_height - row_starts).clamp(
                min=0.0, max=float(patch_size)
            )
            col_extents = (content_width - col_starts).clamp(
                min=0.0, max=float(patch_size)
            )
            normalized_y = (
                row_starts + 0.5 * row_extents
            ).clamp(max=float(content_height)) / float(content_height)
            normalized_x = (
                col_starts + 0.5 * col_extents
            ).clamp(max=float(content_width)) / float(content_width)
            grid_y, grid_x = torch_module.meshgrid(
                normalized_y, normalized_x, indexing="ij"
            )
            sampling_grid = torch_module.stack(
                (2.0 * grid_x - 1.0, 2.0 * grid_y - 1.0), dim=-1
            )
            page_positions = torch_module.nn.functional.grid_sample(
                page_positions,
                sampling_grid.unsqueeze(0),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).to(dtype=page.dtype)
        else:
            page_positions = torch_module.nn.functional.interpolate(
                positional_grid.float(),
                size=(rows, cols),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(dtype=page.dtype)
        page_positions = page_positions.flatten(2).permute(0, 2, 1)
        positioned_pages.append(page + page_positions)
    if cursor != int(patches.shape[1]):
        raise ValueError(
            f"Normalized position split consumed {cursor} of {int(patches.shape[1])} patches"
        )
    return torch_module.cat(positioned_pages, dim=1)


def _patch_nemotron_parse_navit() -> None:
    try:
        import inspect
        from functools import partial
        from typing import Annotated, Literal

        import torch
        import vllm.model_executor.models.nemotron_parse as np
        from vllm.model_executor.models.radio import RadioModel
        from vllm.multimodal.inputs import MultiModalFieldConfig
        from vllm.multimodal.processing import PromptReplacement
        from vllm.utils.tensor_schema import TensorSchema, TensorShape
    except Exception:
        return

    # The compatibility module enables postponed annotations.
    # TensorSchema resolves annotations later with the module globals, so expose
    # the optional vLLM typing symbols there rather than leaving them local.
    globals().update(
        Annotated=Annotated,
        Literal=Literal,
        torch=torch,
        TensorShape=TensorShape,
    )

    model_cls = getattr(np, "NemotronParseForConditionalGeneration", None)
    neck_cls = getattr(np, "RadioWithNeck", None)
    processor_cls = getattr(np, "NemotronParseMultiModalProcessor", None)
    if model_cls is None or neck_cls is None or processor_cls is None:
        return
    if getattr(model_cls, "_np_navit_patch", False):
        return
    if "imgs_sizes" not in inspect.signature(RadioModel.forward).parameters:
        raise RuntimeError(
            "Nemotron Parse NaViT export requires a vLLM RADIO build with imgs_sizes support"
        )
    if not hasattr(MultiModalFieldConfig, "flat_from_sizes"):
        raise RuntimeError(
            "Nemotron Parse NaViT export requires MultiModalFieldConfig.flat_from_sizes"
        )

    class NemotronParseNavitPixelInputs(TensorSchema):
        type: Literal["pixel_values"]
        data: Annotated[torch.Tensor, TensorShape("np", "cps")]
        image_sizes: Annotated[torch.Tensor, TensorShape("ni", 2)]
        content_sizes: Annotated[torch.Tensor, TensorShape("ni", 2)]

    def get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        del self, hf_processor_mm_kwargs
        image_sizes = hf_inputs.get("image_sizes", torch.empty((0, 2)))
        patch_size = 16
        patch_counts = (image_sizes[:, 0] // patch_size) * (
            image_sizes[:, 1] // patch_size
        )
        return {
            "pixel_values": MultiModalFieldConfig.flat_from_sizes(
                "image", patch_counts
            ),
            "image_sizes": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            "content_sizes": MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
        }

    def get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        del self, mm_items, hf_processor_mm_kwargs

        def get_replacement(item_idx: int):
            size = out_mm_kwargs["image"][item_idx]["image_sizes"].data
            height, width = (int(value) for value in size.reshape(-1).tolist())
            patch_cols = width // 16
            pooled_cols = (
                (patch_cols + _NP_NECK_COMPRESSION - 1) // _NP_NECK_COMPRESSION
                if _NP_MASKED_PARTIAL_POOL
                else patch_cols // _NP_NECK_COMPRESSION
            )
            return [0] * ((height // 16) * pooled_cols + 1)

        return [
            PromptReplacement(
                modality="image",
                target=[0],
                replacement=get_replacement,
            )
        ]

    def parse_image_input(self, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        content_sizes = kwargs.pop("content_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None and image_embeds is not None:
            raise ValueError("Both pixel values and image embeds are provided")
        if image_embeds is not None:
            raise NotImplementedError("NaViT image_embeds input is not implemented")
        if image_sizes is None:
            raise ValueError("NaViT pixel_values require image_sizes")
        if (
            _NP_NAVIT_POSITION_EMBEDDING_MODE
            == "content_normalized_2d_interpolate"
            and content_sizes is None
        ):
            raise ValueError(
                "content_normalized_2d_interpolate requires content_sizes"
            )
        return NemotronParseNavitPixelInputs(
            type="pixel_values",
            data=pixel_values,
            image_sizes=image_sizes,
            content_sizes=content_sizes,
        )

    def packed_neck_forward(
        self, pixel_values, *, image_sizes=None, content_sizes=None, **kwargs
    ):
        del kwargs
        if image_sizes is None:
            raise ValueError("Packed Nemotron Parse RADIO requires image_sizes")
        if pixel_values.ndim == 2:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 3 or pixel_values.shape[0] != 1:
            raise ValueError(
                f"Expected packed raw patches [1,np,cps], got {tuple(pixel_values.shape)}"
            )
        sizes = [tuple(int(value) for value in row) for row in image_sizes]
        normalized_content_sizes = (
            [tuple(int(value) for value in row) for row in content_sizes]
            if content_sizes is not None
            else None
        )
        patch_counts = [(height // 16) * (width // 16) for height, width in sizes]
        # RadioModel._extract_final currently applies summary_idxs once to the
        # full packed sequence, which drops summaries for later images. Run the
        # packed transformer itself, then split each page's prefix and patches.
        radio = self.model_encoder.model
        patch_generator = radio.patch_generator
        virtual_offsets = []
        normalized_modes = {
            "normalized_2d_interpolate",
            "content_normalized_2d_interpolate",
        }
        if _NP_NAVIT_POSITION_EMBEDDING_MODE in normalized_modes:
            if _NP_NAVIT_COORDINATE_FRAME == "legacy_virtual_canvas":
                raise ValueError(
                    "normalized_2d_interpolate cannot use legacy_virtual_canvas"
                )
            patches = patch_generator.embedder(pixel_values)
            patches = _normalized_2d_position_pages(
                torch,
                patches=patches,
                pos_embed=patch_generator.pos_embed,
                source_rows=patch_generator.num_rows,
                source_cols=patch_generator.num_cols,
                image_sizes=sizes,
                patch_size=patch_generator.patch_size,
                content_sizes=normalized_content_sizes,
                content_normalized=(
                    _NP_NAVIT_POSITION_EMBEDDING_MODE
                    == "content_normalized_2d_interpolate"
                ),
            )
            patches = patch_generator.cls_token_dynamic(patches, imgs_sizes=sizes)
            hidden_states = patch_generator.patch_normalizer(patches)
            mask_meta = (
                radio.inter_image_mask_metadata(sizes, device=hidden_states.device)
                if len(sizes) > 1
                else None
            )
            hidden = radio.encoder(inputs_embeds=hidden_states, mask_meta=mask_meta)
            virtual_offsets = [(0, 0)] * len(sizes)
        elif _NP_NAVIT_POSITION_EMBEDDING_MODE != "radio_cpe":
            raise ValueError(
                "Unsupported NaViT position embedding mode: "
                f"{_NP_NAVIT_POSITION_EMBEDDING_MODE}"
            )
        elif _NP_NAVIT_COORDINATE_FRAME == "legacy_virtual_canvas":
            # The processor already emitted flattened, normalized 16x16 raw
            # patches. Embed only those patches, but gather CPE positions from
            # the 2048x1664 legacy lattice before inserting per-page prefixes.
            patches = patch_generator.embedder(pixel_values)
            canvas_height, canvas_width = 2048, 1664
            canvas_cols = canvas_width // 16
            all_patch_idxs = []
            for height, width in sizes:
                left = ((canvas_width - width) // 2 // 64) * 64
                top = ((canvas_height - height) // 2 // 16) * 16
                if left < 0 or top < 0:
                    raise ValueError(
                        f"Native page {(height, width)} exceeds legacy virtual canvas"
                    )
                rows, cols = height // 16, width // 16
                row0, col0 = top // 16, left // 16
                page_idxs = torch.tensor(
                    [
                        (row0 + row) * canvas_cols + col0 + col
                        for row in range(rows)
                        for col in range(cols)
                    ],
                    dtype=torch.long,
                    device=patches.device,
                )
                all_patch_idxs.append(page_idxs)
                virtual_offsets.append((top, left))
            patch_idxs = torch.cat(all_patch_idxs).unsqueeze(0)
            pos_result = patch_generator.apply_pos_enc(
                patches,
                patch_idxs=patch_idxs,
                input_size=(canvas_height, canvas_width),
            )
            patches = pos_result[0] if isinstance(pos_result, tuple) else pos_result
            patches = patch_generator.cls_token_dynamic(patches, imgs_sizes=sizes)
            hidden_states = patch_generator.patch_normalizer(patches)
            mask_meta = (
                radio.inter_image_mask_metadata(sizes, device=hidden_states.device)
                if len(sizes) > 1
                else None
            )
            hidden = radio.encoder(inputs_embeds=hidden_states, mask_meta=mask_meta)
        else:
            hidden = radio(pixel_values, imgs_sizes=sizes)
            virtual_offsets = [(0, 0)] * len(sizes)
        num_skip = int(patch_generator.num_skip)
        num_cls_tokens = int(patch_generator.num_cls_tokens)
        summary_idxs = self.model_encoder.summary_idxs
        summaries = []
        features = []
        cursor = 0
        for patch_count in patch_counts:
            page_summary = hidden[:, cursor : cursor + num_cls_tokens]
            if summary_idxs is not None:
                page_summary = page_summary[:, summary_idxs]
            summaries.append(page_summary.flatten(1))
            features.append(
                hidden[:, cursor + num_skip : cursor + num_skip + patch_count]
            )
            cursor += num_skip + patch_count
        if cursor != hidden.shape[1]:
            raise ValueError(
                f"Packed RADIO split consumed {cursor} of {hidden.shape[1]} tokens"
            )
        outputs = []
        for (height, width), page_summary, page_feature in zip(
            sizes, summaries, features
        ):
            output = self.conv1(page_feature.permute(0, 2, 1)).permute(0, 2, 1)
            output = self.layer_norm1(output)
            output = np.rearrange(
                output,
                "b (h w) d -> b d h w",
                h=height // 16,
                w=width // 16,
            )
            if _NP_NECK_COMPRESSION == 4 and not _NP_MASKED_PARTIAL_POOL:
                output = self.conv2(output)
            else:
                import torch.nn.functional as F

                kernel = self.conv2.weight
                source_width = int(kernel.shape[-1])
                if _NP_NECK_COMPRESSION % source_width:
                    raise ValueError(
                        f"neck compression {_NP_NECK_COMPRESSION} must be a multiple "
                        f"of trained kernel width {source_width}"
                    )
                repeats = _NP_NECK_COMPRESSION // source_width
                if repeats > 1:
                    kernel = kernel.repeat(1, 1, 1, repeats) / repeats
                patch_cols = int(output.shape[-1])
                pad_cols = (-patch_cols) % _NP_NECK_COMPRESSION
                if pad_cols:
                    if not _NP_MASKED_PARTIAL_POOL:
                        raise ValueError(
                            f"grid width {patch_cols} is not divisible by neck "
                            f"compression {_NP_NECK_COMPRESSION}"
                        )
                    output = F.pad(output, (0, pad_cols, 0, 0))
                output = F.conv2d(
                    output,
                    kernel,
                    bias=self.conv2.bias,
                    stride=(1, _NP_NECK_COMPRESSION),
                )
            output = np.rearrange(output, "b d h w -> b (h w) d")
            output = self.layer_norm2(output)
            projected_summary = self.sum_proj(page_summary)
            if isinstance(projected_summary, tuple):
                projected_summary = projected_summary[0]
            page_summary = self.layer_norm3(projected_summary)
            outputs.append(torch.cat((output, page_summary.unsqueeze(1)), dim=1).squeeze(0))
        log_count = int(getattr(self, "_np_navit_log_count", 0))
        # vLLM consumes early calls while profiling its encoder cache. Keep
        # enough bounded audit lines to also capture real mixed-size requests.
        if log_count < 8:
            post_neck_lengths = [int(output.shape[0]) for output in outputs]
            print(
                "[np-vllm-navit-pack] "
                f"backbone={_NP_BACKBONE} attention={_NP_ATTENTION_MODE} "
                f"vitdet_window={_NP_VITDET_WINDOW or None} "
                f"vitdet_global_layers={_NP_VITDET_GLOBAL_LAYERS} "
                f"neck_compression={_NP_NECK_COMPRESSION} pages={len(sizes)} "
                f"packed_boundaries={list(zip([0] + list(torch.tensor(patch_counts).cumsum(0)[:-1].tolist()), torch.tensor(patch_counts).cumsum(0).tolist()))} "
                f"image_sizes={sizes} "
                f"content_sizes={normalized_content_sizes} "
                f"coordinate_frame={_NP_NAVIT_COORDINATE_FRAME} "
                f"position_embedding_mode={_NP_NAVIT_POSITION_EMBEDDING_MODE} "
                f"virtual_offsets={virtual_offsets} "
                f"raw_patches={patch_counts} post_neck_lengths={post_neck_lengths}",
                flush=True,
            )
            self._np_navit_log_count = log_count + 1
        return tuple(outputs)

    def process_image_input(self, image_input):
        pixel_values = image_input["data"]
        image_sizes = image_input["image_sizes"]
        content_sizes = image_input["content_sizes"]
        dtype = next(self.encoder.parameters()).dtype
        return self.encoder(
            pixel_values.to(dtype),
            image_sizes=image_sizes.reshape(-1, 2).tolist(),
            content_sizes=(
                content_sizes.reshape(-1, 2).tolist()
                if content_sizes is not None
                else None
            ),
        )

    processor_cls._get_mm_fields_config = get_mm_fields_config
    processor_cls._get_prompt_updates = get_prompt_updates
    neck_cls.forward = packed_neck_forward
    model_cls._parse_and_validate_image_input = parse_image_input
    model_cls._process_image_input = process_image_input
    model_cls._np_navit_patch = True
    print(
        "[np-vllm-navit-patch] enabled dynamic rectangles, packed RADIO attention, "
        f"coordinate_frame={_NP_NAVIT_COORDINATE_FRAME}, backbone={_NP_BACKBONE}, "
        f"attention={_NP_ATTENTION_MODE}, vitdet_window={_NP_VITDET_WINDOW or None}, "
        f"neck_compression={_NP_NECK_COMPRESSION}, "
        f"masked_partial_pool={_NP_MASKED_PARTIAL_POOL}",
        flush=True,
    )


_patch_nemotron_parse_navit()


"""Runtime GQA/MQA adapter for vLLM 0.26 Nemotron Parse mBART."""

import os


def _patch_mbart_gqa() -> None:
    try:
        from torch import nn
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            QKVParallelLinear,
            ColumnParallelLinear,
            RowParallelLinear,
        )
        import vllm.model_executor.models.nemotron_parse as np
        from vllm.model_executor.models.whisper import (
            CrossAttention,
            WhisperAttention,
            WhisperCrossAttention,
        )
        from vllm.v1.attention.backend import AttentionType
    except Exception:
        return

    self_kv_heads = int(os.environ.get("NP_VLLM_SELF_KV_HEADS", "8"))
    cross_kv_heads = int(os.environ.get("NP_VLLM_CROSS_KV_HEADS", "16"))

    def local_kv_heads(total: int) -> int:
        tp_size = get_tensor_model_parallel_world_size()
        if total >= tp_size:
            if total % tp_size:
                raise ValueError(f"KV heads {total} must be divisible by TP {tp_size}")
            return total // tp_size
        if tp_size % total:
            raise ValueError(f"TP {tp_size} must be divisible by KV heads {total}")
        return 1

    class NemotronParseGQASelfAttention(WhisperAttention):
        def __init__(
            self, embed_dim, num_heads, bias=True,
            attn_type=AttentionType.DECODER, per_layer_sliding_window=None,
            cache_config=None, quant_config=None, prefix="",
        ):
            # Do not call WhisperAttention.__init__: it registers a full-MHA
            # Attention object under this prefix before we can supply the GQA
            # cache shape, and vLLM correctly rejects the duplicate layer.
            nn.Module.__init__(self)
            self.embed_dim = embed_dim
            self.total_num_heads = num_heads
            tp_size = get_tensor_model_parallel_world_size()
            if num_heads % tp_size or num_heads % self_kv_heads:
                raise ValueError("self KV heads must divide query heads")
            self.num_heads = num_heads // tp_size
            self.num_kv_heads = local_kv_heads(self_kv_heads)
            self.head_dim = embed_dim // num_heads
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.attn_type = attn_type
            self.scaling = self.head_dim**-0.5
            self.qkv_proj = QKVParallelLinear(
                hidden_size=self.embed_dim,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self_kv_heads,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.out_proj = RowParallelLinear(
                input_size=embed_dim, output_size=embed_dim, bias=bias,
                quant_config=quant_config, prefix=f"{prefix}.out_proj",
            )
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
                per_layer_sliding_window=per_layer_sliding_window,
            )

    class NemotronParseGQACrossAttention(WhisperCrossAttention):
        def __init__(
            self, embed_dim, num_heads, bias=True, cache_config=None,
            quant_config=None, prefix="",
        ):
            nn.Module.__init__(self)
            self.embed_dim = embed_dim
            self.total_num_heads = num_heads
            tp_size = get_tensor_model_parallel_world_size()
            if num_heads % tp_size or num_heads % cross_kv_heads:
                raise ValueError("cross KV heads must divide query heads")
            self.num_heads = num_heads // tp_size
            self.num_kv_heads = local_kv_heads(cross_kv_heads)
            self.head_dim = embed_dim // num_heads
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.attn_type = AttentionType.ENCODER_DECODER
            self.scaling = self.head_dim**-0.5
            output_size = cross_kv_heads * self.head_dim
            self.q_proj = ColumnParallelLinear(
                input_size=embed_dim, output_size=embed_dim, bias=bias,
                quant_config=quant_config, prefix=f"{prefix}.q_proj",
            )
            self.kv_proj = MergedColumnParallelLinear(
                input_size=self.embed_dim,
                output_sizes=[output_size, output_size],
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_proj",
            )
            self.out_proj = RowParallelLinear(
                input_size=embed_dim, output_size=embed_dim, bias=bias,
                quant_config=quant_config, prefix=f"{prefix}.out_proj",
            )
            self.attn = CrossAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=self.attn_type,
            )

    np.WhisperAttention = NemotronParseGQASelfAttention
    np.WhisperCrossAttention = NemotronParseGQACrossAttention
    print(
        "[np-vllm-gqa] "
        f"self_q_heads=16 self_kv_heads={self_kv_heads} "
        f"cross_q_heads=16 cross_kv_heads={cross_kv_heads}",
        flush=True,
    )
    print("[np-vllm-arch] mBART grouped-query attention enabled", flush=True)


_patch_mbart_gqa()


"""Register the local same-weight mBART visual-prefix architecture."""


def _register_radio_mbart_prefix():
    try:
        from vllm.model_executor.models import ModelRegistry
        from vllm.model_executor.models.interfaces import supports_multimodal
        try:
            from .vllm_radio_mbart_prefix import RadioMBartPrefixForCausalLM
        except ImportError:
            # Legacy PYTHONPATH mode, retained for existing launch contracts.
            from vllm_radio_mbart_prefix import RadioMBartPrefixForCausalLM
    except Exception:
        return
    ModelRegistry.register_model("RadioMBartPrefixForCausalLM", RadioMBartPrefixForCausalLM)
    if not supports_multimodal(RadioMBartPrefixForCausalLM):
        raise RuntimeError("RadioMBartPrefixForCausalLM did not register as multimodal")
    print("[np-vllm-mbart-prefix] same decoder weights; cross-attention bypassed", flush=True)
    print("[np-vllm-arch] mBART causal visual-prefix fusion enabled", flush=True)


_register_radio_mbart_prefix()


"""Model-scoped default for the validated FA3 scheduler path."""


def _patch_vllm_fa3_aot_scheduler() -> None:
    """Bypass FA3 AOT scheduler metadata by default for Nemotron Parse.

    vLLM 0.26 can size the full-CUDA-graph scheduler metadata incorrectly for
    a mixed batch containing long prefills and cached one-token decodes. Keep
    FlashAttention 3 and CUDA graphs enabled while bypassing that optional
    schedule, without enabling vLLM's broader batch-invariant execution mode.
    """
    try:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadataBuilder,
        )
    except Exception:
        return

    original_init = FlashAttentionMetadataBuilder.__init__
    if getattr(original_init, "_np_fa3_no_aot_patch", False):
        return

    def init_without_aot_schedule(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        from .defaults import disable_fa3_aot

        if not disable_fa3_aot(self.model_config):
            return
        self.aot_schedule = False
        self.scheduler_metadata = None
        self.max_num_splits = 0
        print("[nemotron-parse] FA3 AOT scheduler disabled for Parse", flush=True)

    init_without_aot_schedule._np_fa3_no_aot_patch = True
    FlashAttentionMetadataBuilder.__init__ = init_without_aot_schedule
    print(
        "[np-vllm-fa3] registered Parse FA3 scheduler defaults; FA3 remains enabled",
        flush=True,
    )


_patch_vllm_fa3_aot_scheduler()
