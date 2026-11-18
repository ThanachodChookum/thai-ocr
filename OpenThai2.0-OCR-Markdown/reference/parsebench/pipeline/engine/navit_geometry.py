# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Coordinate transforms for native-resolution NaViT image grids.

``legacy_virtual_canvas`` removes blank pixels from compute while preserving
the legacy 1664x2048 coordinate and positional-embedding lattice.  A native
rectangle is assigned an aligned location on that virtual canvas; targets,
RADIO patch positions, and inference postprocessing all use the same location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


LEGACY_CANVAS_WIDTH = 1664
LEGACY_CANVAS_HEIGHT = 2048
NAVIT_WIDTH_MULTIPLE = 64
NAVIT_HEIGHT_MULTIPLE = 16
LEGACY_VIRTUAL_CANVAS_FRAME = "legacy_virtual_canvas"
SUPPORTED_NAVIT_COORDINATE_FRAMES = {
    "padded_grid",
    "content_normalized",
    LEGACY_VIRTUAL_CANVAS_FRAME,
}


def _round_up(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return ((max(1, int(value)) + multiple - 1) // multiple) * multiple


def top_left_navit_grid(
    width: int,
    height: int,
    *,
    width_multiple: int = 64,
    height_multiple: int = 16,
) -> Tuple[int, int, int, int]:
    """Return padded width/height and deterministic top-left offsets."""

    return (
        _round_up(width, width_multiple),
        _round_up(height, height_multiple),
        0,
        0,
    )


def aligned_virtual_canvas_offsets(
    width: int,
    height: int,
    *,
    canvas_width: int = LEGACY_CANVAS_WIDTH,
    canvas_height: int = LEGACY_CANVAS_HEIGHT,
    width_multiple: int = NAVIT_WIDTH_MULTIPLE,
    height_multiple: int = NAVIT_HEIGHT_MULTIPLE,
) -> Tuple[int, int]:
    """Center a native grid on the legacy canvas without breaking neck groups.

    The RADIO patch lattice is 16px.  The neck groups four adjacent columns,
    so the horizontal origin must also be a multiple of 64px.  Rounding the
    centered offset down gives deterministic placement and leaves any one-patch
    asymmetry on the bottom/right, matching normal padding conventions.
    """

    width, height = int(width), int(height)
    canvas_width, canvas_height = int(canvas_width), int(canvas_height)
    if width <= 0 or height <= 0:
        raise ValueError("native grid dimensions must be positive")
    if width > canvas_width or height > canvas_height:
        raise ValueError(
            f"native grid {(width, height)} does not fit virtual canvas "
            f"{(canvas_width, canvas_height)}"
        )
    left = ((canvas_width - width) // 2 // int(width_multiple)) * int(width_multiple)
    top = ((canvas_height - height) // 2 // int(height_multiple)) * int(height_multiple)
    return left, top


def virtual_canvas_patch_indices(
    width: int,
    height: int,
    *,
    pad_left: int,
    pad_top: int,
    canvas_width: int = LEGACY_CANVAS_WIDTH,
    canvas_height: int = LEGACY_CANVAS_HEIGHT,
    patch_size: int = 16,
) -> Tuple[int, ...]:
    """Return row-major full-canvas patch indices for a native rectangle."""

    values = (width, height, pad_left, pad_top, canvas_width, canvas_height)
    if any(int(value) % int(patch_size) for value in values):
        raise ValueError("virtual-canvas geometry must be patch aligned")
    if pad_left < 0 or pad_top < 0:
        raise ValueError("virtual-canvas offsets must be non-negative")
    if pad_left + width > canvas_width or pad_top + height > canvas_height:
        raise ValueError("native grid does not fit inside virtual canvas")
    canvas_cols = int(canvas_width) // int(patch_size)
    local_cols = int(width) // int(patch_size)
    local_rows = int(height) // int(patch_size)
    col0 = int(pad_left) // int(patch_size)
    row0 = int(pad_top) // int(patch_size)
    return tuple(
        (row0 + row) * canvas_cols + col0 + col
        for row in range(local_rows)
        for col in range(local_cols)
    )


def padded_bbox_to_virtual_canvas_normalized(
    bbox: Sequence[float],
    *,
    virtual_pad_left: int,
    virtual_pad_top: int,
    canvas_width: int = LEGACY_CANVAS_WIDTH,
    canvas_height: int = LEGACY_CANVAS_HEIGHT,
    clip: bool = True,
) -> Tuple[float, float, float, float]:
    """Place a bbox from the physical native grid on the virtual canvas."""

    values = (
        (float(bbox[0]) + virtual_pad_left) / canvas_width,
        (float(bbox[1]) + virtual_pad_top) / canvas_height,
        (float(bbox[2]) + virtual_pad_left) / canvas_width,
        (float(bbox[3]) + virtual_pad_top) / canvas_height,
    )
    if not clip:
        return values
    return tuple(min(1.0, max(0.0, value)) for value in values)


def padded_bbox_to_content_normalized(
    bbox: Sequence[float],
    *,
    content_width: int,
    content_height: int,
    pad_left: int = 0,
    pad_top: int = 0,
    clip: bool = True,
) -> Tuple[float, float, float, float]:
    """Remove grid padding and normalize a box by the visible content frame."""

    if content_width <= 0 or content_height <= 0:
        raise ValueError("content dimensions must be positive")
    values = (
        (float(bbox[0]) - pad_left) / content_width,
        (float(bbox[1]) - pad_top) / content_height,
        (float(bbox[2]) - pad_left) / content_width,
        (float(bbox[3]) - pad_top) / content_height,
    )
    if not clip:
        return values
    return tuple(min(1.0, max(0.0, value)) for value in values)


def quantize_normalized_coordinate(value: float, bins: int) -> float:
    """Quantize to the existing coordinate-token grid while excluding 1.0."""

    bins = int(bins)
    if bins <= 0:
        raise ValueError("coordinate bins must be positive")
    index = min(bins - 1, max(0, int(float(value) * bins)))
    return round(index / bins, 4)


@dataclass(frozen=True)
class NavitImageGeometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    padded_width: int
    padded_height: int
    pad_left: int = 0
    pad_top: int = 0
    coordinate_frame: str = "padded_grid"

    @classmethod
    def from_original_size(
        cls,
        original_width: int,
        original_height: int,
        *,
        max_width: int = 1664,
        max_height: int = 2048,
        width_multiple: int = 64,
        height_multiple: int = 16,
        pad_left: int = 0,
        pad_top: int = 0,
        coordinate_frame: str = "padded_grid",
    ) -> "NavitImageGeometry":
        """Reproduce the HF processor's aspect resize and NaViT grid pad."""

        original_width = int(original_width)
        original_height = int(original_height)
        if original_width <= 0 or original_height <= 0:
            raise ValueError("original image dimensions must be positive")

        aspect_ratio = original_width / original_height
        resized_width = original_width
        resized_height = original_height
        if resized_height > int(max_height):
            resized_height = int(max_height)
            resized_width = int(resized_height * aspect_ratio)
        if resized_width > int(max_width):
            resized_width = int(max_width)
            resized_height = int(resized_width / aspect_ratio)

        padded_width = _round_up(resized_width, width_multiple)
        padded_height = _round_up(resized_height, height_multiple)
        if pad_left < 0 or pad_top < 0:
            raise ValueError("padding offsets must be non-negative")
        if pad_left + resized_width > padded_width or pad_top + resized_height > padded_height:
            raise ValueError("resized content does not fit inside the padded grid")
        return cls(
            original_width=original_width,
            original_height=original_height,
            resized_width=resized_width,
            resized_height=resized_height,
            padded_width=padded_width,
            padded_height=padded_height,
            pad_left=int(pad_left),
            pad_top=int(pad_top),
            coordinate_frame=str(coordinate_frame),
        )

    @classmethod
    def from_fixed_canvas(
        cls,
        original_width: int,
        original_height: int,
        *,
        canvas_width: int = 1664,
        canvas_height: int = 2048,
    ) -> "NavitImageGeometry":
        """Reproduce the historical aspect resize and centered fixed canvas."""

        geometry = cls.from_original_size(
            original_width,
            original_height,
            max_width=canvas_width,
            max_height=canvas_height,
            width_multiple=1,
            height_multiple=1,
        )
        pad_left = (int(canvas_width) - geometry.resized_width) // 2
        pad_top = (int(canvas_height) - geometry.resized_height) // 2
        return cls(
            original_width=geometry.original_width,
            original_height=geometry.original_height,
            resized_width=geometry.resized_width,
            resized_height=geometry.resized_height,
            padded_width=int(canvas_width),
            padded_height=int(canvas_height),
            pad_left=pad_left,
            pad_top=pad_top,
            coordinate_frame="padded_grid",
        )

    @classmethod
    def from_legacy_virtual_canvas(
        cls,
        original_width: int,
        original_height: int,
        *,
        canvas_width: int = LEGACY_CANVAS_WIDTH,
        canvas_height: int = LEGACY_CANVAS_HEIGHT,
    ) -> "NavitImageGeometry":
        """Build the native-grid/virtual-canvas inference geometry contract."""

        native = cls.from_original_size(
            original_width,
            original_height,
            max_width=canvas_width,
            max_height=canvas_height,
            width_multiple=NAVIT_WIDTH_MULTIPLE,
            height_multiple=NAVIT_HEIGHT_MULTIPLE,
        )
        virtual_left, virtual_top = aligned_virtual_canvas_offsets(
            native.padded_width,
            native.padded_height,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        return cls(
            original_width=native.original_width,
            original_height=native.original_height,
            resized_width=native.resized_width,
            resized_height=native.resized_height,
            padded_width=int(canvas_width),
            padded_height=int(canvas_height),
            pad_left=virtual_left + native.pad_left,
            pad_top=virtual_top + native.pad_top,
            coordinate_frame=LEGACY_VIRTUAL_CANVAS_FRAME,
        )

    def original_bbox_to_model(self, bbox: Sequence[float]) -> Tuple[float, float, float, float]:
        """Map an original-pixel bbox to the configured normalized frame."""

        if self.coordinate_frame == "content_normalized":
            return (
                float(bbox[0]) / self.original_width,
                float(bbox[1]) / self.original_height,
                float(bbox[2]) / self.original_width,
                float(bbox[3]) / self.original_height,
            )
        if self.coordinate_frame not in {"padded_grid", LEGACY_VIRTUAL_CANVAS_FRAME}:
            raise ValueError(f"unsupported coordinate frame: {self.coordinate_frame}")

        x_scale = self.resized_width / self.original_width
        y_scale = self.resized_height / self.original_height
        return (
            (float(bbox[0]) * x_scale + self.pad_left) / self.padded_width,
            (float(bbox[1]) * y_scale + self.pad_top) / self.padded_height,
            (float(bbox[2]) * x_scale + self.pad_left) / self.padded_width,
            (float(bbox[3]) * y_scale + self.pad_top) / self.padded_height,
        )

    def model_bbox_to_original(
        self,
        bbox: Sequence[float],
        *,
        clip: bool = True,
    ) -> Tuple[float, float, float, float]:
        """Map a normalized model-frame bbox back to original image pixels."""

        if self.coordinate_frame == "content_normalized":
            values = (
                float(bbox[0]) * self.original_width,
                float(bbox[1]) * self.original_height,
                float(bbox[2]) * self.original_width,
                float(bbox[3]) * self.original_height,
            )
            if not clip:
                return values
            return (
                min(self.original_width, max(0.0, values[0])),
                min(self.original_height, max(0.0, values[1])),
                min(self.original_width, max(0.0, values[2])),
                min(self.original_height, max(0.0, values[3])),
            )
        if self.coordinate_frame not in {"padded_grid", LEGACY_VIRTUAL_CANVAS_FRAME}:
            raise ValueError(f"unsupported coordinate frame: {self.coordinate_frame}")

        values = (
            (float(bbox[0]) * self.padded_width - self.pad_left)
            * self.original_width
            / self.resized_width,
            (float(bbox[1]) * self.padded_height - self.pad_top)
            * self.original_height
            / self.resized_height,
            (float(bbox[2]) * self.padded_width - self.pad_left)
            * self.original_width
            / self.resized_width,
            (float(bbox[3]) * self.padded_height - self.pad_top)
            * self.original_height
            / self.resized_height,
        )
        if not clip:
            return values
        return (
            min(self.original_width, max(0.0, values[0])),
            min(self.original_height, max(0.0, values[1])),
            min(self.original_width, max(0.0, values[2])),
            min(self.original_height, max(0.0, values[3])),
        )
