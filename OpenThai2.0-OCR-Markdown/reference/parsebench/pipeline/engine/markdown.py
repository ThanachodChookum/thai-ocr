# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import json


import math


import multiprocessing as mp


import os


import re


import sys


import tarfile


from collections import Counter, defaultdict


from concurrent.futures import ProcessPoolExecutor


from dataclasses import dataclass


from datetime import datetime


from pathlib import Path


from typing import Any, Iterable


NEMOTRON_LATEX2HTML_ROOT = Path(".")


TEXT_BEARING_CLASSES = {
    "Text",
    "Title",
    "Section-header",
    "List-item",
    "TOC",
    "Bibliography",
    "Footnote",
    "Caption",
    "Chart",
    "Code",
    "Page-header",
    "Page-footer",
}


from .blocks import BLOCK_RE as _RE_EXTRACT_CLASS_BBOX, text_blocks


def extract_classes_bboxes(text: str) -> tuple[list[str], list[tuple[float, float, float, float]], list[str]]:
    classes: list[str] = []
    bboxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    for match in _RE_EXTRACT_CLASS_BBOX.finditer(text or ""):
        x1, y1, body, x2, y2, cls = match.groups()
        classes.append(cls)
        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        texts.append(body)
    return classes, bboxes, texts


def convert_latex_tables_in_markdown(markdown_text: str) -> str:
    if "\\begin{tabular}" not in markdown_text:
        return markdown_text
    try:
        from .convert.latex2html import latex_table_to_html
        return latex_table_to_html(markdown_text)
    except Exception:
        return _simple_latex_table_to_html(markdown_text)


def _simple_latex_table_to_html(text: str) -> str:
    pattern = re.compile(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", re.DOTALL)

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        rows = []
        for row in body.split("\\\\"):
            row = re.sub(r"\\hline|\\toprule|\\midrule|\\bottomrule", "", row).strip()
            if not row:
                continue
            cells = [re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", c).strip() for c in row.split("&")]
            cells = ["" if re.fullmatch(r"\{\s*\}", c) else c for c in cells]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        return "<table>" + "".join(rows) + "</table>"

    return pattern.sub(repl, text)


# Text remains usable when a coordinate token is malformed. Keep numerical
# coordinate validation in extract_classes_bboxes for callers that need geometry.



def mmd_to_markdown(prediction: str, text_transform=None) -> str:
    parsed = list(text_blocks(prediction or ""))
    classes = [cls for cls, _ in parsed]
    texts = [text for _, text in parsed]
    if not texts:
        # No class-terminated block was completed. Keep the original response
        # available to callers instead of presenting a partial table as parsed.
        if any(token in (prediction or "") for token in ['<x_', '<y_']):
            return ""
        cleaned = re.sub(r"<x_[^>]+>|<y_[^>]+>|<class_[^>]+>", "", prediction or "")
        return convert_latex_tables_in_markdown(cleaned.replace("<tbc>", "").strip())

    blocks = []
    for cls, text in zip(classes, texts):
        if text_transform is not None:text=text_transform(cls,text)
        text = text.replace("<tbc>", "").strip()
        if not text:
            continue
        if cls == "Table":
            text = convert_latex_tables_in_markdown(text)
        elif cls == "Title" and not text.startswith("#"):
            text = f"# {text}"
        elif cls == "Section-header" and not text.startswith("#"):
            text = f"## {text}"
        blocks.append(text)
    return convert_latex_tables_in_markdown("\n\n".join(blocks).strip())

