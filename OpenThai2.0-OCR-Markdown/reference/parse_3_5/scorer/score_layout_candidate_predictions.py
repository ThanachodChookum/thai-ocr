#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Score Nemotron Parse predictions on a standalone layout validation split.

Evaluate Parse-style block tags against bundled JSONL labels. By default,
Table bodies use the installed release's HTML conversion before scoring.
Use --table-mode legacy for the historical standard-library-only scorer.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


BLOCK_RE = re.compile(
    r"<x_([+-]?\d+(?:\.\d+)?)><y_([+-]?\d+(?:\.\d+)?)>(.*?)"
    r"<x_([+-]?\d+(?:\.\d+)?)><y_([+-]?\d+(?:\.\d+)?)><class_([^>]+)>",
    re.MULTILINE | re.DOTALL,
)
TAG_RE = re.compile(r"<(?:x|y)_[^>]+>|<class_[^>]+>|<tbc>|</?s>")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Bundle labels.jsonl.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction JSONL path(s); globs are allowed.")
    parser.add_argument("--out", type=Path, required=True, help="Summary metrics JSON.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional per-page CSV.")
    parser.add_argument("--canvas-width", type=float, default=1664.0)
    parser.add_argument("--canvas-height", type=float, default=2048.0)
    parser.add_argument(
        "--coord-mode",
        choices=["centerpad", "normalized", "absolute"],
        default="centerpad",
        help="How to decode predicted coordinates. centerpad matches Nemotron Parse preprocessing.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--table-match-iou", type=float, default=0.25)
    parser.add_argument(
        "--table-mode", choices=["released", "legacy"], default="released",
        help="released converts predicted Table bodies with nemotron_parse; legacy preserves historical raw scoring.",
    )
    return parser.parse_args()


def convert_predicted_tables(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    """Render only Table bodies; keep boxes/classes/other text intact.

    Ground truth never enters this function. Preserve malformed table source
    when the released converter cannot render it; keep such outputs in scoring.
    """
    try:
        from nemotron_parse.postprocessing import PostprocessOptions, postprocess_text
    except ImportError as exc:
        raise SystemExit(
            "Released table scoring requires the model repository package: "
            "python -m pip install /path/to/model. "
            "For historical raw scoring use --table-mode legacy."
        ) from exc
    counts = {"table_blocks": 0, "changed_blocks": 0, "unconverted_latex_blocks": 0}

    def render(text: str) -> str:
        counts["table_blocks"] += 1
        converted = postprocess_text(
            text, "Table", table_format="HTML",
            options=PostprocessOptions(render_charts=False),
        )
        counts["changed_blocks"] += int(converted != text)
        counts["unconverted_latex_blocks"] += int(r"\begin{tabular" in converted)
        return converted

    def replace(match: re.Match[str]) -> str:
        if match.group(6).casefold() != "table":
            return match[0]
        start, end = match.span(3)
        offset = match.start()
        return match[0][:start-offset] + render(match[3]) + match[0][end-offset:]

    result = dict(row)
    raw_key = next((key for key in ("raw_out", "prediction", "text") if row.get(key)), None)
    raw = row.get(raw_key, "") if raw_key else ""
    if raw and BLOCK_RE.search(raw):
        result[raw_key] = BLOCK_RE.sub(replace, raw)
    else:
        result["blocks"] = []
        for block in row.get("blocks") or []:
            converted = dict(block)
            category = block.get("category") or block.get("class") or block.get("label")
            if class_group(category) == "table":
                converted["text"] = render(block.get("text") or "")
            result["blocks"].append(converted)
    return result, counts


class SimpleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            text = normalize_text("".join(self._cell))
            if self._row is None:
                self._row = []
            self._row.append(text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None


def expand_prediction_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return paths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def prediction_name(row: dict[str, Any]) -> str | None:
    if row.get("image_name"):
        return Path(str(row["image_name"])).name
    if row.get("image_path"):
        return Path(str(row["image_path"])).name
    return None


def normalize_text(value: str) -> str:
    value = html.unescape(value or "")
    value = TAG_RE.sub(" ", value)
    value = HTML_TAG_RE.sub(" ", value)
    value = re.sub(r"[*_`#]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip().casefold()


def strip_model_markup(value: str) -> str:
    return normalize_text(value)


def char_bag_f1(reference: str, prediction: str) -> float | None:
    ref = Counter(ch for ch in normalize_text(reference) if not ch.isspace())
    pred = Counter(ch for ch in normalize_text(prediction) if not ch.isspace())
    if not ref and not pred:
        return None
    if not ref or not pred:
        return 0.0
    overlap = sum((ref & pred).values())
    precision = overlap / sum(pred.values()) if pred else 0.0
    recall = overlap / sum(ref.values()) if ref else 0.0
    return f1(precision, recall)


def cell_bag_f1(reference_rows: list[list[str]], prediction_rows: list[list[str]]) -> float | None:
    ref = Counter(normalize_text(cell) for row in reference_rows for cell in row if normalize_text(cell))
    pred = Counter(normalize_text(cell) for row in prediction_rows for cell in row if normalize_text(cell))
    if not ref and not pred:
        return None
    if not ref or not pred:
        return 0.0
    overlap = sum((ref & pred).values())
    precision = overlap / sum(pred.values()) if pred else 0.0
    recall = overlap / sum(ref.values()) if ref else 0.0
    return f1(precision, recall)


def f1(precision: float, recall: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def parse_html_table(value: str) -> list[list[str]]:
    if "<table" not in value and "<tr" not in value:
        return []
    parser = SimpleTableParser()
    parser.feed(value)
    return parser.rows


def parse_markdown_table(value: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if "|" not in line:
            continue
        trimmed = line.strip("|").strip()
        if not trimmed:
            continue
        cells = [normalize_text(cell) for cell in trimmed.split("|")]
        if not any(cells):
            continue
        if all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells if cell):
            continue
        rows.append(cells)
    return rows


def parse_latex_table(value: str) -> list[list[str]]:
    if "\\begin{tabular" not in value:
        return []
    match = re.search(r"\\begin\{tabular\}.*?}(.*?)\\end\{tabular\}", value, re.DOTALL)
    body = match.group(1) if match else value
    rows = []
    for row_text in re.split(r"\\\\", body):
        row_text = re.sub(r"\\(?:hline|toprule|midrule|bottomrule)\b", " ", row_text)
        row_text = re.sub(r"\\(?:multicolumn|multirow)\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", row_text)
        cells = [normalize_text(re.sub(r"\\[A-Za-z]+", " ", cell)) for cell in row_text.split("&")]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(cells)
    return rows


def parse_table_rows(value: str) -> list[list[str]]:
    for parser in (parse_html_table, parse_markdown_table, parse_latex_table):
        rows = parser(value or "")
        if rows:
            return rows
    return []


def grid_similarity(reference_rows: list[list[str]], prediction_rows: list[list[str]]) -> float | None:
    if not reference_rows and not prediction_rows:
        return None
    if not reference_rows or not prediction_rows:
        return 0.0
    ref_r = len(reference_rows)
    pred_r = len(prediction_rows)
    ref_c = max((len(row) for row in reference_rows), default=0)
    pred_c = max((len(row) for row in prediction_rows), default=0)
    row_sim = 1.0 - abs(ref_r - pred_r) / max(ref_r, pred_r, 1)
    col_sim = 1.0 - abs(ref_c - pred_c) / max(ref_c, pred_c, 1)
    return max(0.0, (row_sim + col_sim) / 2.0)


def class_group(value: str | None) -> str:
    raw = (value or "").casefold()
    raw = raw.replace("_", "-").replace(" ", "-")
    compact = re.sub(r"[^a-z0-9-]+", "", raw)
    compact = compact.strip("-")
    aliases = {
        "pageheader": "page-header",
        "page-header": "page-header",
        "header": "page-header",
        "pagefooter": "page-footer",
        "page-footer": "page-footer",
        "footer": "page-footer",
        "page-number": "page-footer",
        "pagenumber": "page-footer",
        "sectionheader": "title",
        "section-header": "title",
        "title": "title",
        "caption": "caption",
        "footnote": "caption",
        "figure-caption": "caption",
        "table-caption": "caption",
        "text": "text",
        "paragraph": "text",
        "list-item": "text",
        "listitem": "text",
        "reference": "text",
        "table": "table",
        "picture": "picture",
        "figure": "picture",
        "image": "picture",
        "chart": "picture",
        "formula": "formula",
        "equation": "formula",
    }
    return aliases.get(compact, compact or "unknown")


def decode_coord(value: float, canvas_axis: float) -> float:
    if abs(value) > 2.0:
        return value / canvas_axis
    return value


def clip_bbox(coords: list[float], *, width: float, height: float) -> list[float]:
    x1, y1, x2, y2 = coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        max(0.0, min(width, x1)),
        max(0.0, min(height, y1)),
        max(0.0, min(width, x2)),
        max(0.0, min(height, y2)),
    ]


def decode_bbox(
    coords: list[float],
    *,
    width: float,
    height: float,
    canvas_width: float,
    canvas_height: float,
    coord_mode: str,
) -> list[float]:
    x1, y1, x2, y2 = coords
    if coord_mode == "absolute":
        abs_box = [x1, y1, x2, y2]
    else:
        x1n = decode_coord(x1, canvas_width)
        y1n = decode_coord(y1, canvas_height)
        x2n = decode_coord(x2, canvas_width)
        y2n = decode_coord(y2, canvas_height)
        if coord_mode == "normalized":
            abs_box = [x1n * width, y1n * height, x2n * width, y2n * height]
        else:
            scale = min(canvas_width / width, canvas_height / height)
            resized_width = width * scale
            resized_height = height * scale
            pad_x = (canvas_width - resized_width) / 2.0
            pad_y = (canvas_height - resized_height) / 2.0
            abs_box = [
                (x1n * canvas_width - pad_x) / scale,
                (y1n * canvas_height - pad_y) / scale,
                (x2n * canvas_width - pad_x) / scale,
                (y2n * canvas_height - pad_y) / scale,
            ]
    return clip_bbox(abs_box, width=width, height=height)


def extract_pred_blocks(row: dict[str, Any], label: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    width = float(label.get("width") or 1)
    height = float(label.get("height") or 1)
    raw_out = row.get("raw_out") or row.get("prediction") or row.get("text") or ""
    blocks: list[dict[str, Any]] = []
    for match in BLOCK_RE.finditer(raw_out):
        x1, y1, text, x2, y2, cls = match.groups()
        bbox = decode_bbox(
            [float(x1), float(y1), float(x2), float(y2)],
            width=width,
            height=height,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            coord_mode=args.coord_mode,
        )
        blocks.append(
            {
                "bbox": bbox,
                "category": cls,
                "category_group": class_group(cls),
                "text": text.replace("<tbc>", "").strip(),
            }
        )
    if blocks:
        return blocks

    for block in row.get("blocks") or []:
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox = decode_bbox(
            [float(v) for v in bbox],
            width=width,
            height=height,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            coord_mode=args.coord_mode,
        )
        category = block.get("category") or block.get("class") or block.get("label")
        blocks.append(
            {
                "bbox": bbox,
                "category": category,
                "category_group": class_group(category),
                "text": block.get("text") or "",
            }
        )
    return blocks


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = area([ix1, iy1, ix2, iy2])
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def maximum_cardinality_matches(
    gt_blocks: list[dict[str, Any]],
    pred_blocks: list[dict[str, Any]],
    threshold: float,
    *,
    require_same_group: bool = False,
) -> list[tuple[int, int, float]]:
    """Return the largest valid one-to-one IoU matching.

    A greedy highest-IoU assignment can leave a matchable ground-truth block
    unmatched.  The small validation sets do not need scipy: an augmenting-path
    bipartite matcher gives the correct maximum cardinality while considering
    higher-IoU edges first for deterministic tie breaking.
    """
    adjacency: list[list[tuple[int, float]]] = []
    for gt in gt_blocks:
        candidates = []
        for pred_i, pred in enumerate(pred_blocks):
            if require_same_group and gt["category_group"] != pred["category_group"]:
                continue
            value = iou(gt["bbox"], pred["bbox"])
            if value >= threshold:
                candidates.append((pred_i, value))
        adjacency.append(sorted(candidates, key=lambda item: (-item[1], item[0])))

    pred_to_gt: dict[int, int] = {}

    def augment(gt_i: int, seen_pred: set[int]) -> bool:
        for pred_i, _ in adjacency[gt_i]:
            if pred_i in seen_pred:
                continue
            seen_pred.add(pred_i)
            previous_gt = pred_to_gt.get(pred_i)
            if previous_gt is None or augment(previous_gt, seen_pred):
                pred_to_gt[pred_i] = gt_i
                return True
        return False

    for gt_i in sorted(range(len(gt_blocks)), key=lambda index: (len(adjacency[index]), index)):
        augment(gt_i, set())

    matches = []
    for pred_i, gt_i in pred_to_gt.items():
        value = iou(gt_blocks[gt_i]["bbox"], pred_blocks[pred_i]["bbox"])
        matches.append((gt_i, pred_i, value))
    return sorted(matches)


def greedy_matches(gt_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]], threshold: float) -> list[tuple[int, int, float]]:
    """Backward-compatible name for the maximum-cardinality matcher."""
    return maximum_cardinality_matches(gt_blocks, pred_blocks, threshold)


def same_group_matches(gt_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]], threshold: float) -> list[tuple[int, int, float]]:
    return maximum_cardinality_matches(
        gt_blocks,
        pred_blocks,
        threshold,
        require_same_group=True,
    )


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def best_iou_per_gt(gt_blocks: list[dict[str, Any]], pred_blocks: list[dict[str, Any]]) -> list[float]:
    values = []
    for gt in gt_blocks:
        values.append(max((iou(gt["bbox"], pred["bbox"]) for pred in pred_blocks), default=0.0))
    return values


def score_table_content(
    label: dict[str, Any],
    raw_out: str,
    gt_blocks: list[dict[str, Any]],
    pred_blocks: list[dict[str, Any]],
    *,
    table_match_iou: float,
) -> dict[str, Any]:
    table_blocks = [block for block in gt_blocks if block["category_group"] == "table"]
    pred_table_blocks = [block for block in pred_blocks if block["category_group"] == "table"]
    if not table_blocks:
        return {
            "table_count": 0,
            "table_text_boc_f1": None,
            "table_cell_bag_f1": None,
            "table_grid_similarity": None,
            "table_score": None,
        }

    table_matches = maximum_cardinality_matches(
        table_blocks,
        pred_table_blocks,
        table_match_iou,
    )
    pred_by_gt = {gt_i: pred_i for gt_i, pred_i, _ in table_matches}

    text_scores: list[float | None] = []
    cell_scores: list[float | None] = []
    grid_scores: list[float | None] = []
    for gt_i, gt in enumerate(table_blocks):
        gt_text = gt.get("html") or gt.get("text") or ""
        pred_i = pred_by_gt.get(gt_i)
        pred_text = pred_table_blocks[pred_i].get("text") or "" if pred_i is not None else ""
        gt_rows = parse_table_rows(gt_text)
        pred_rows = parse_table_rows(pred_text)
        text_scores.append(char_bag_f1(gt_text, pred_text))
        cell_scores.append(cell_bag_f1(gt_rows, pred_rows))
        grid_scores.append(grid_similarity(gt_rows, pred_rows))

    table_text = mean(text_scores)
    table_cell = mean(cell_scores)
    table_grid = mean(grid_scores)
    components = [value for value in (table_text, table_cell, table_grid) if value is not None]
    return {
        "table_count": len(table_blocks),
        "table_text_boc_f1": table_text,
        "table_cell_bag_f1": table_cell,
        "table_grid_similarity": table_grid,
        "table_score": sum(components) / len(components) if components else None,
    }


def score_page(label: dict[str, Any], pred: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    raw_out = (pred or {}).get("raw_out") or (pred or {}).get("prediction") or (pred or {}).get("text") or ""
    width = float(label.get("width") or 1)
    height = float(label.get("height") or 1)
    gt_blocks = [
        {
            "bbox": clip_bbox(
                [float(v) for v in block["bbox"]],
                width=width,
                height=height,
            ),
            "category": block["category"],
            "category_group": block["category_group"],
            "text": block.get("text") or "",
            "html": block.get("html") or "",
        }
        for block in label.get("blocks", [])
    ]
    pred_blocks = extract_pred_blocks(pred or {}, label, args) if pred else []
    loc_matches = greedy_matches(gt_blocks, pred_blocks, args.iou_threshold)
    class_matches = same_group_matches(gt_blocks, pred_blocks, args.iou_threshold)

    class_recalls: dict[str, float | None] = {}
    for group in sorted({block["category_group"] for block in gt_blocks} | {block["category_group"] for block in pred_blocks}):
        gt_sub = [block for block in gt_blocks if block["category_group"] == group]
        pred_sub = [block for block in pred_blocks if block["category_group"] == group]
        matches = greedy_matches(gt_sub, pred_sub, args.iou_threshold)
        class_recalls[f"{group}_recall_iou50"] = len(matches) / len(gt_sub) if gt_sub else None

    loc_precision = len(loc_matches) / len(pred_blocks) if pred_blocks else (0.0 if gt_blocks else 1.0)
    loc_recall = len(loc_matches) / len(gt_blocks) if gt_blocks else None
    class_precision = len(class_matches) / len(pred_blocks) if pred_blocks else (0.0 if gt_blocks else 1.0)
    class_recall = len(class_matches) / len(gt_blocks) if gt_blocks else None
    page_metrics: dict[str, Any] = {
        "image_name": label["image_name"],
        "bucket": label.get("bucket"),
        "prediction_found": pred is not None,
        "gt_blocks": len(gt_blocks),
        "pred_blocks": len(pred_blocks),
        "mean_best_iou": mean(best_iou_per_gt(gt_blocks, pred_blocks)),
        "localization_recall_iou50": loc_recall,
        "localization_precision_iou50": loc_precision,
        "class_accuracy_iou50": len(class_matches) / len(loc_matches) if loc_matches else None,
        "class_recall_iou50": class_recall,
        "class_precision_iou50": class_precision,
        "class_f1_iou50": f1(class_precision or 0.0, class_recall or 0.0)
        if class_precision is not None and class_recall is not None
        else None,
        "text_boc_f1": char_bag_f1(label.get("ground_truth_text") or "", raw_out),
    }
    page_metrics.update(class_recalls)
    page_metrics.update(
        score_table_content(
            label,
            raw_out,
            gt_blocks,
            pred_blocks,
            table_match_iou=args.table_match_iou,
        )
    )
    return page_metrics


def flatten_metrics(page_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({key for row in page_rows for key in row if key not in {"image_name", "bucket", "prediction_found"}})
    flat: dict[str, Any] = {
        "num_pages": len(page_rows),
        "num_predictions_found": sum(1 for row in page_rows if row.get("prediction_found")),
    }
    for name in metric_names:
        values = [row.get(name) for row in page_rows if isinstance(row.get(name), (int, float))]
        if values:
            flat[f"layout_lite/{name}"] = sum(float(value) for value in values) / len(values)
    table_pages = [row for row in page_rows if row.get("table_count", 0) > 0]
    if table_pages:
        for name in ("table_text_boc_f1", "table_cell_bag_f1", "table_grid_similarity", "table_score"):
            values = [row.get(name) for row in table_pages if isinstance(row.get(name), (int, float))]
            if values:
                flat[f"layout_lite/{name}_table_pages"] = sum(float(value) for value in values) / len(values)

    overall_parts = [
        flat.get("layout_lite/class_f1_iou50"),
        flat.get("layout_lite/table_score_table_pages"),
        flat.get("layout_lite/picture_recall_iou50"),
    ]
    overall_values = [float(value) for value in overall_parts if isinstance(value, (int, float))]
    flat["layout_lite/overall"] = sum(overall_values) / len(overall_values) if overall_values else None
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    labels = load_jsonl(args.labels)
    labels_by_image = {row["image_name"]: row for row in labels}

    pred_paths = expand_prediction_paths(args.predictions)
    predictions: dict[str, dict[str, Any]] = {}
    for path in pred_paths:
        if not path.is_file():
            raise SystemExit(f"Missing prediction file: {path}")
        for row in load_jsonl(path):
            name = prediction_name(row)
            if name:
                predictions[name] = row

    conversion_counts = {"table_blocks": 0, "changed_blocks": 0, "unconverted_latex_blocks": 0}
    if args.table_mode == "released":
        for name, row in predictions.items():
            predictions[name], counts = convert_predicted_tables(row)
            for key, value in counts.items():
                conversion_counts[key] += value

    page_rows = [score_page(label, predictions.get(label["image_name"]), args) for label in labels]
    flat = flatten_metrics(page_rows)
    missing = [label["image_name"] for label in labels if label["image_name"] not in predictions]
    extra = sorted(set(predictions) - set(labels_by_image))
    result = {
        "metric_definition": {
            "name": "layout_lite",
            "version": 3 if args.table_mode == "released" else 2,
            "prediction_table_processing": args.table_mode,
            "aggregation": "macro_per_page",
            "matching": "maximum_cardinality_one_to_one",
            "unmatched_table_content_score": 0.0,
            "empty_prediction_precision_with_ground_truth": 0.0,
            "ground_truth_boxes_clipped_to_image": True,
            "overall_components": [
                "class_f1_iou50",
                "table_score_table_pages",
                "picture_recall_iou50",
            ],
        },
        "labels": str(args.labels),
        "predictions": [str(path) for path in pred_paths],
        "coordinate_mode": args.coord_mode,
        "iou_threshold": args.iou_threshold,
        "table_match_iou": args.table_match_iou,
        "table_conversion": conversion_counts,
        "num_labels": len(labels),
        "num_predictions": len(predictions),
        "missing_predictions": missing,
        "extra_predictions": extra,
        "flat_metrics": flat,
        "per_page": page_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.csv_out is not None:
        write_csv(args.csv_out, page_rows)
    print(json.dumps(flat, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
