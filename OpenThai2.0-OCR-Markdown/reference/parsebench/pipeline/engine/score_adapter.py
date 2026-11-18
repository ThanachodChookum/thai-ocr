# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from .navit_geometry import NavitImageGeometry
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


DEFAULT_PARSEBENCH_ROOT = Path(".")


DEFAULT_PARSEBENCH_SRC = Path(".")


NEMOTRON_LATEX2HTML_ROOT = Path(".")


DEFAULT_MODEL_CANVAS_WIDTH = 1664.0


DEFAULT_MODEL_CANVAS_HEIGHT = 2048.0


LEADERBOARD_METRICS = {
    "table": "grits_trm_composite",
    "chart": "rule_pass_rate",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
    "layout": "layout_element_rule_pass_rate",
}


LEADERBOARD_FIELDS = {
    "table": "tables",
    "chart": "charts",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
    "layout": "visual_grounding",
}


YOLO_LABELS = {
    "Caption": "0",
    "Footnote": "1",
    "Formula": "2",
    "List-item": "3",
    "ListItem": "3",
    "Page-footer": "4",
    "PageFooter": "4",
    "Page-header": "5",
    "PageHeader": "5",
    "Picture": "6",
    "Figure": "6",
    "Chart": "6",
    "Section-header": "7",
    "SectionHeader": "7",
    "Table": "8",
    "Text": "9",
    "Title": "10",
    # Non-Core classes are mapped to the closest text-like label for eval.
    "Code": "9",
    "TOC": "9",
    "Bibliography": "9",
}


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


_RE_EXTRACT_CLASS_BBOX = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


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


@dataclass
class ParseBenchSample:
    key: str
    data: dict[str, Any]

    @property
    def category(self) -> str:
        return str(self.data.get("category", ""))

    @property
    def test_id(self) -> str:
        stem = Path(str(self.data.get("pdf_rel") or self.key)).stem
        group = self.inference_group
        return str(self.data.get("test_id") or f"{group}/{stem}")

    @property
    def inference_group(self) -> str:
        if self.category in {"text_content", "text_formatting"}:
            return "text"
        return self.category or "parsebench"

    @property
    def pdf_rel(self) -> str:
        return str(self.data.get("pdf_rel") or self.data.get("pdf") or "")

    @property
    def expected_markdown(self) -> str:
        value = self.data.get("expected_markdown") or self.data.get("ground_truth")
        if value:
            return str(value)
        block_text = self.data.get("block_text")
        if isinstance(block_text, list):
            return "\n\n".join(str(x) for x in block_text)
        return ""

    @property
    def page_width(self) -> int:
        page_info = self.data.get("page_info") or {}
        return int(page_info.get("width") or self.data.get("width") or 1664)

    @property
    def page_height(self) -> int:
        page_info = self.data.get("page_info") or {}
        return int(page_info.get("height") or self.data.get("height") or 2048)

    @property
    def rule_meta(self) -> dict[str, Any]:
        meta = dict(self.data.get("rule_meta") or {})
        for rule in self.all_rules:
            for key in ("allow_splitting_ambiguous_merged_tables", "trm_unsupported", "max_top_title_rows"):
                if key in rule and key not in meta:
                    meta[key] = rule[key]
        return meta

    @property
    def all_rules(self) -> list[dict[str, Any]]:
        rules = self.data.get("rules", [])
        return [r for r in rules if isinstance(r, dict)]

    @property
    def pointer_rules(self) -> list[dict[str, Any]]:
        return [r for r in self.all_rules if r.get("type") == "expected_markdown"]

    @property
    def parse_rules_by_category(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rule in self.all_rules:
            rule_type = rule.get("type")
            if rule_type in {"expected_markdown", "layout"}:
                continue
            category = str(rule.get("category") or self.category)
            grouped[category].append(rule)
        return dict(grouped)

    @property
    def layout_rules(self) -> list[dict[str, Any]]:
        return [r for r in self.all_rules if r.get("type") == "layout"]


class ParseBenchCache:
    def __init__(self, data_root: Path = DEFAULT_PARSEBENCH_ROOT):
        self.data_root = Path(data_root)
        self.samples: dict[str, ParseBenchSample] = {}
        self._load()

    def _load(self) -> None:
        train_dir = self.data_root / "train"
        if not train_dir.exists():
            return
        for tar_path in sorted(train_dir.glob("*.tar")):
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if not member.name.endswith(".json"):
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    data = json.loads(fh.read())
                    key = member.name.rsplit(".", 1)[0]
                    sample = ParseBenchSample(key=key, data=data)
                    aliases = {
                        key,
                        Path(key).name,
                        f"{tar_path.parent.name}/{tar_path.name}/{Path(key).name}",
                        f"{tar_path.name}/{Path(key).name}",
                    }
                    if sample.test_id:
                        aliases.add(sample.test_id)
                    for alias in aliases:
                        self.samples[alias] = sample

    def resolve(self, key: str) -> ParseBenchSample | None:
        key = str(key)
        return (
            self.samples.get(key)
            or self.samples.get(Path(key).name)
            or self.samples.get(key.split("/")[-1])
            or self.samples.get(key.split("\\")[-1])
        )

    def inventory(self) -> dict[str, Any]:
        seen: dict[str, ParseBenchSample] = {sample.key: sample for sample in self.samples.values()}
        page_categories = Counter(sample.category for sample in seen.values())
        rule_categories: Counter[str] = Counter()
        rule_types: Counter[str] = Counter()
        expected_markdown = 0
        for sample in seen.values():
            if sample.expected_markdown:
                expected_markdown += 1
            for rule in sample.all_rules:
                rule_categories[str(rule.get("category") or sample.category)] += 1
                rule_types[str(rule.get("type") or "")] += 1
        return {
            "samples": len(seen),
            "page_categories": dict(page_categories),
            "rule_categories": dict(rule_categories),
            "rule_types": dict(rule_types),
            "expected_markdown_samples": expected_markdown,
        }


def ensure_parsebench_imports(parsebench_src: Path = DEFAULT_PARSEBENCH_SRC) -> None:
    import enum

    if not hasattr(enum, "StrEnum"):
        class StrEnum(str, enum.Enum):
            pass

        enum.StrEnum = StrEnum
    src = str(parsebench_src)
    if src not in sys.path:
        sys.path.insert(0, src)


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


def mmd_to_markdown(prediction: str) -> str:
    classes, _boxes, texts = extract_classes_bboxes(prediction or "")
    if not texts:
        cleaned = re.sub(r"<x_[^>]+>|<y_[^>]+>|<class_[^>]+>", "", prediction or "")
        return convert_latex_tables_in_markdown(cleaned.replace("<tbc>", "").strip())

    blocks = []
    for cls, text in zip(classes, texts):
        text = text.replace("<tbc>", "").strip()
        if not text:
            continue
        if cls == "Table":
            text = convert_latex_tables_in_markdown(text)
        elif cls == "Title" and not text.startswith("#"):
            text = f"# {text}"
        elif cls == "Section-header" and not text.startswith("#"):
            text = f"## {text}"
        elif cls not in TEXT_BEARING_CLASSES and cls not in {"Formula", "Picture"}:
            continue
        blocks.append(text)
    return convert_latex_tables_in_markdown("\n\n".join(blocks).strip())


def _clean_block_text(cls: str, text: str) -> str:
    text = text.replace("<tbc>", "").strip()
    if cls == "Table":
        return convert_latex_tables_in_markdown(text)
    if cls == "Title" and text and not text.startswith("#"):
        return f"# {text}"
    if cls == "Section-header" and text and not text.startswith("#"):
        return f"## {text}"
    return text


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _model_canvas_size() -> tuple[float, float]:
    width = _env_float("PARSEBENCH_MODEL_CANVAS_WIDTH", _env_float("EVAL_CANVAS_WIDTH", DEFAULT_MODEL_CANVAS_WIDTH))
    height = _env_float("PARSEBENCH_MODEL_CANVAS_HEIGHT", _env_float("EVAL_CANVAS_HEIGHT", DEFAULT_MODEL_CANVAS_HEIGHT))
    return max(1.0, width), max(1.0, height)


def _unpad_prediction_boxes() -> bool:
    value = os.environ.get("PARSEBENCH_UNPAD_PRED_BBOXES", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _prediction_pad_position() -> str:
    value = os.environ.get("PARSEBENCH_PRED_PAD_POSITION", "center").strip().lower()
    if value in {"top-left", "topleft", "top_left"}:
        return "top-left"
    if value in {"bottom-right", "bottomright", "bottom_right"}:
        return "bottom-right"
    return "center"


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _bbox_to_page_normalized(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    coordinate_frame: str | None = None,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 1.5:
        norm = (x1, y1, x2, y2)
    elif max_coord <= 1000.0:
        norm = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
    else:
        norm = (x1 / max(1, width), y1 / max(1, height), x2 / max(1, width), y2 / max(1, height))

    virtual_canvas = coordinate_frame == "legacy_virtual_canvas"
    if coordinate_frame in {"content_normalized", "original_normalized", "navit_content"}:
        unpad_prediction = False
    elif coordinate_frame in {"padded_grid", "fixed_canvas", "navit_grid", "legacy_virtual_canvas"}:
        unpad_prediction = True
    elif coordinate_frame is None:
        # Backward compatibility for old JSONL records without provenance.
        unpad_prediction = _unpad_prediction_boxes()
    else:
        raise ValueError(f"Unsupported prediction coordinate frame: {coordinate_frame!r}")

    # Padded-grid box tokens are normalized after resize+pad. ParseBench expects
    # boxes in the original page frame. Content-normalized predictions already
    # use that frame and must never be unpadded a second time.
    if unpad_prediction:
        if virtual_canvas:
            geometry = NavitImageGeometry.from_legacy_virtual_canvas(width, height)
            original = geometry.model_bbox_to_original(norm, clip=True)
            norm = (
                original[0] / max(1, width),
                original[1] / max(1, height),
                original[2] / max(1, width),
                original[3] / max(1, height),
            )
            left, right = sorted((_clamp01(norm[0]), _clamp01(norm[2])))
            top, bottom = sorted((_clamp01(norm[1]), _clamp01(norm[3])))
            return left, top, right, bottom
        canvas_width, canvas_height = _model_canvas_size()
        # Eval preprocessing only downsizes oversized pages, then pads to the
        # fixed model canvas.  Do not invert a fictional upscale for pages that
        # were already smaller than the canvas.
        scale = min(1.0, canvas_width / max(1, width), canvas_height / max(1, height))
        resized_width = width * scale
        resized_height = height * scale
        pad_position = _prediction_pad_position()
        if pad_position == "top-left":
            pad_x = 0.0
            pad_y = 0.0
        elif pad_position == "bottom-right":
            pad_x = canvas_width - resized_width
            pad_y = canvas_height - resized_height
        else:
            pad_x = (canvas_width - resized_width) / 2.0
            pad_y = (canvas_height - resized_height) / 2.0
        x1 = (norm[0] * canvas_width - pad_x) / max(1.0, resized_width)
        y1 = (norm[1] * canvas_height - pad_y) / max(1.0, resized_height)
        x2 = (norm[2] * canvas_width - pad_x) / max(1.0, resized_width)
        y2 = (norm[3] * canvas_height - pad_y) / max(1.0, resized_height)
        norm = (x1, y1, x2, y2)

    left, right = sorted((_clamp01(norm[0]), _clamp01(norm[2])))
    top, bottom = sorted((_clamp01(norm[1]), _clamp01(norm[3])))
    return left, top, right, bottom


def _bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    coordinate_frame: str | None = None,
) -> list[float]:
    x1, y1, x2, y2 = _bbox_to_page_normalized(bbox, width, height, coordinate_frame)
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def _bbox_to_xywh_normalized(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    coordinate_frame: str | None = None,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = _bbox_to_page_normalized(bbox, width, height, coordinate_frame)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _label_to_yolo(cls: str) -> str:
    return YOLO_LABELS.get(cls, YOLO_LABELS.get(cls.replace("_", "-"), "9"))


def _layout_item_type(cls: str) -> str:
    if cls == "Table":
        return "table"
    if cls in {"Picture", "Figure", "Chart"}:
        return "image"
    if cls == "Formula":
        return "formula"
    if cls == "Title":
        return "heading"
    return "text"


def to_parse_output(
    prediction: str,
    sample: ParseBenchSample,
    pipeline_name: str = "nemotron_parse",
    coordinate_frame: str | None = None,
):
    ensure_parsebench_imports()
    from parse_bench.schemas.parse_output import LayoutItemIR, LayoutSegmentIR, PageIR, ParseLayoutPageIR, ParseOutput

    markdown = mmd_to_markdown(prediction)
    classes, boxes, texts = extract_classes_bboxes(prediction or "")
    items = []
    headers = []
    footers = []
    for order_index, (cls, bbox, text) in enumerate(zip(classes, boxes, texts)):
        x, y, w, h = _bbox_to_xywh_normalized(
            bbox, sample.page_width, sample.page_height, coordinate_frame
        )
        segment = LayoutSegmentIR(
            x=x,
            y=y,
            w=w,
            h=h,
            confidence=1.0,
            label=cls,
            start_index=None,
            end_index=None,
        )
        content = _clean_block_text(cls, text)
        if cls == "Page-header" and content:
            headers.append(content)
        elif cls == "Page-footer" and content:
            footers.append(content)
        items.append(
            LayoutItemIR(
                type=_layout_item_type(cls),
                md=content,
                html=content if "<table" in content.lower() else "",
                value=content,
                bbox=segment,
                layout_segments=[segment],
            )
        )
    return ParseOutput(
        example_id=sample.test_id,
        pipeline_name=pipeline_name,
        pages=[PageIR(page_index=0, markdown=markdown)],
        layout_pages=[
            ParseLayoutPageIR(
                page_number=1,
                width=float(sample.page_width),
                height=float(sample.page_height),
                md=markdown,
                text=markdown,
                page_header_markdown="\n\n".join(headers),
                page_footer_markdown="\n\n".join(footers),
                items=items,
            )
        ],
        markdown=markdown,
    )


def to_layout_output(
    prediction: str,
    sample: ParseBenchSample,
    pipeline_name: str = "nemotron_parse",
    coordinate_frame: str | None = None,
):
    ensure_parsebench_imports()
    from parse_bench.schemas.layout_detection_output import (
        LayoutDetectionModel,
        LayoutOutput,
        LayoutPrediction,
        LayoutTableContent,
        LayoutTextContent,
    )

    markdown = mmd_to_markdown(prediction)
    classes, boxes, texts = extract_classes_bboxes(prediction or "")
    predictions = []
    for cls, bbox, text in zip(classes, boxes, texts):
        html_or_text = _clean_block_text(cls, text)
        content = None
        if cls == "Table" and "<table" in html_or_text.lower():
            content = LayoutTableContent(html=html_or_text)
        elif cls in TEXT_BEARING_CLASSES:
            content = LayoutTextContent(text=html_or_text)
        predictions.append(
            LayoutPrediction(
                bbox=_bbox_to_pixels(
                    bbox, sample.page_width, sample.page_height, coordinate_frame
                ),
                score=1.0,
                label=_label_to_yolo(cls),
                page=1,
                content=content,
                provider_metadata={"source_label": cls},
            )
        )
    return LayoutOutput(
        example_id=sample.test_id,
        pipeline_name=pipeline_name,
        model=LayoutDetectionModel.YOLO_DOCLAYNET,
        image_width=sample.page_width,
        image_height=sample.page_height,
        predictions=predictions,
        markdown=markdown,
    )


def make_inference_result(
    prediction: str,
    sample: ParseBenchSample,
    *,
    product_type: str = "parse",
    pipeline_name: str = "nemotron_parse",
    coordinate_frame: str | None = None,
):
    ensure_parsebench_imports()
    from parse_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
    from parse_bench.schemas.product import ProductType

    now = datetime.now()
    if product_type == "layout_detection":
        output = to_layout_output(prediction, sample, pipeline_name, coordinate_frame)
        ptype = ProductType.LAYOUT_DETECTION
    else:
        output = to_parse_output(prediction, sample, pipeline_name, coordinate_frame)
        ptype = ProductType.PARSE
    request = InferenceRequest(
        example_id=sample.test_id,
        source_file_path=sample.pdf_rel or sample.key,
        product_type=ptype,
    )
    return InferenceResult(
        request=request,
        pipeline_name=pipeline_name,
        product_type=ptype,
        raw_output={
            "markdown": mmd_to_markdown(prediction),
            "source_format": "eclair_mmd",
        },
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )


def make_parse_test_case(sample: ParseBenchSample, category: str, rules: list[dict[str, Any]] | None = None):
    ensure_parsebench_imports()
    from parse_bench.test_cases.schema import ParseTestCase

    meta = sample.rule_meta
    expected = sample.expected_markdown if category == "table" else None
    return ParseTestCase(
        test_id=f"{category}/{Path(sample.pdf_rel or sample.key).stem}",
        group=category,
        file_path=Path(sample.pdf_rel or sample.key),
        tags=[category],
        test_rules=rules or None,
        expected_markdown=expected,
        allow_splitting_ambiguous_merged_tables=bool(meta.get("allow_splitting_ambiguous_merged_tables", False)),
        trm_unsupported=bool(meta.get("trm_unsupported", False)),
        max_top_title_rows=int(meta.get("max_top_title_rows", 1)),
    )


def make_layout_test_case(sample: ParseBenchSample):
    ensure_parsebench_imports()
    from parse_bench.test_cases.schema import LayoutDetectionTestCase

    return LayoutDetectionTestCase(
        test_id=f"layout/{Path(sample.pdf_rel or sample.key).stem}",
        group="layout",
        file_path=Path(sample.pdf_rel or sample.key),
        tags=["layout"],
        test_rules=sample.layout_rules,
        ontology="basic",
        page_index=0,
    )


def _metric_dict(metrics: Iterable[Any]) -> dict[str, float]:
    out = {}
    for metric in metrics:
        value = getattr(metric, "value", None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[str(metric.metric_name)] = float(value)
    return out


def _metric_metadata(metrics: Iterable[Any]) -> dict[str, Any]:
    out = {}
    for metric in metrics:
        out[str(metric.metric_name)] = getattr(metric, "metadata", {}) or {}
    return out


class OfficialParseBenchScorer:
    def __init__(self, data_root: Path = DEFAULT_PARSEBENCH_ROOT, pipeline_name: str = "nemotron_parse"):
        ensure_parsebench_imports()
        from parse_bench.evaluation.evaluators.parse import ParseEvaluator

        self.cache = ParseBenchCache(data_root)
        self.pipeline_name = pipeline_name
        self._layout_evaluator = None
        self.parse_evaluator = ParseEvaluator(
            enable_rule_based=True,
            enable_text_similarity=False,
            enable_teds=False,
            enable_grits=True,
            enable_header_accuracy=False,
            enable_structural_consistency=True,
            enable_table_record_match=True,
            enable_table_composite=False,
        )

    @property
    def layout_evaluator(self):
        if self._layout_evaluator is None:
            from parse_bench.evaluation.evaluators.layoutdet import LayoutDetectionEvaluator

            self._layout_evaluator = LayoutDetectionEvaluator()
        return self._layout_evaluator

    def score_record(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        sample = self.cache.resolve(str(record.get("key", "")))
        if sample is None:
            return [{
                "category": "missing",
                "success": False,
                "error": f"missing ParseBench sample for key={record.get('key')}",
                "metrics": {},
                "metadata": {},
            }]

        prediction = str(record.get("raw_out", ""))
        results: list[dict[str, Any]] = []
        coordinate_frame = record.get("coordinate_frame")
        parse_result = make_inference_result(
            prediction,
            sample,
            product_type="parse",
            pipeline_name=self.pipeline_name,
            coordinate_frame=coordinate_frame,
        )

        if sample.expected_markdown and (sample.pointer_rules or sample.category == "table"):
            results.append(self._score_parse_category("table", parse_result, sample, None))

        for category, rules in sorted(sample.parse_rules_by_category.items()):
            if not rules:
                continue
            results.append(self._score_parse_category(category, parse_result, sample, rules))

        if sample.layout_rules:
            layout_result = make_inference_result(
                prediction,
                sample,
                product_type="layout_detection",
                pipeline_name=self.pipeline_name,
                coordinate_frame=coordinate_frame,
            )
            test_case = make_layout_test_case(sample)
            try:
                eval_result = self.layout_evaluator.evaluate(layout_result, test_case)
                results.append({
                    "category": "layout",
                    "success": bool(eval_result.success),
                    "error": eval_result.error,
                    "metrics": _metric_dict(eval_result.metrics),
                    "metadata": _metric_metadata(eval_result.metrics),
                })
            except Exception as exc:
                results.append({
                    "category": "layout",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                    "metadata": {},
                })
        return results

    def _score_parse_category(
        self,
        category: str,
        inference_result: Any,
        sample: ParseBenchSample,
        rules: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        test_case = make_parse_test_case(sample, category, rules)
        try:
            eval_result = self.parse_evaluator.evaluate(inference_result, test_case)
            return {
                "category": category,
                "success": bool(eval_result.success),
                "error": eval_result.error,
                "metrics": _metric_dict(eval_result.metrics),
                "metadata": _metric_metadata(eval_result.metrics),
            }
        except Exception as exc:
            return {
                "category": category,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {},
                "metadata": {},
            }

    def score_records(
        self,
        records: Iterable[dict[str, Any]],
        *,
        workers: int = 1,
    ) -> dict[str, Any]:
        records = list(records)
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        errors: list[dict[str, Any]] = []
        empty_by_category: Counter[str] = Counter()
        worker_count = max(1, min(int(workers), len(records))) if records else 1

        if worker_count == 1:
            scored_records = (
                (
                    record,
                    not mmd_to_markdown(str(record.get("raw_out", ""))).strip(),
                    self.score_record(record),
                )
                for record in records
            )
        else:
            # The scorer owns evaluator/schema objects that are expensive to
            # reconstruct and are not guaranteed to be pickleable.  The CPU
            # scoring jobs run on Linux, so fork the fully initialized scorer
            # once and preserve input order with executor.map.  Ordered result
            # consumption also preserves the prior floating-point aggregation
            # order exactly.
            global _PARSEBENCH_FORKED_SCORER
            _PARSEBENCH_FORKED_SCORER = self
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("fork"),
            )
            scored_records = (
                (record, is_empty, scored)
                for record, (is_empty, scored) in zip(
                    records,
                    executor.map(_score_record_in_forked_worker, records, chunksize=1),
                )
            )

        try:
            for record, is_empty, scored in scored_records:
                for result in scored:
                    category = result["category"]
                    by_category[category].append(result)
                    if is_empty:
                        empty_by_category[category] += 1
                    if not result.get("success", False):
                        errors.append({
                            "key": record.get("key"),
                            "category": category,
                            "error": result.get("error"),
                        })
        finally:
            if worker_count > 1:
                executor.shutdown(wait=True)
                _PARSEBENCH_FORKED_SCORER = None

        return aggregate_official_results(by_category, len(records), empty_by_category, errors)


_PARSEBENCH_FORKED_SCORER: OfficialParseBenchScorer | None = None


def _score_record_in_forked_worker(record: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Score one record using the scorer inherited by a forked CPU worker."""
    if _PARSEBENCH_FORKED_SCORER is None:
        raise RuntimeError("ParseBench forked worker was not initialized")
    is_empty = not mmd_to_markdown(str(record.get("raw_out", ""))).strip()
    return is_empty, _PARSEBENCH_FORKED_SCORER.score_record(record)


def aggregate_official_results(
    by_category: dict[str, list[dict[str, Any]]],
    total_records: int,
    empty_by_category: Counter[str],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    categories: dict[str, Any] = {}
    leaderboard: dict[str, float] = {}
    leaderboard_stats: dict[str, dict[str, float]] = {}
    for category, results in sorted(by_category.items()):
        metric_values: dict[str, list[float]] = defaultdict(list)
        metadata_totals: dict[str, Counter[str]] = defaultdict(Counter)
        for result in results:
            for name, value in result.get("metrics", {}).items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    metric_values[name].append(float(value))
            for metric_name, metadata in result.get("metadata", {}).items():
                if isinstance(metadata, dict):
                    for k in ("passed", "total"):
                        v = metadata.get(k)
                        if isinstance(v, (int, float)):
                            metadata_totals[metric_name][k] += float(v)
        aggregate = {
            f"avg_{name}": sum(values) / len(values)
            for name, values in sorted(metric_values.items())
            if values
        }
        for metric_name, totals in metadata_totals.items():
            if totals.get("total", 0) > 0:
                aggregate[f"micro_{metric_name}"] = totals["passed"] / totals["total"]
        categories[category] = {
            "n": len(results),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "empty_predictions": int(empty_by_category.get(category, 0)),
            "aggregate_metrics": aggregate,
        }
        primary = LEADERBOARD_METRICS.get(category)
        field = LEADERBOARD_FIELDS.get(category)
        if primary and field:
            values = metric_values.get(primary, [])
            if values:
                value_sum = float(sum(values))
                value_count = float(len(values))
                leaderboard_stats[field] = {"sum": value_sum, "count": value_count}
                leaderboard[field] = value_sum / value_count
    vals = [leaderboard[k] for k in ("tables", "charts", "content_faithfulness", "semantic_formatting", "visual_grounding") if k in leaderboard]
    if vals:
        leaderboard["overall"] = sum(vals) / len(vals)
    return {
        "counts": {
            "records": total_records,
            "category_evals": {k: len(v) for k, v in sorted(by_category.items())},
            "errors": len(errors),
        },
        "leaderboard": leaderboard,
        "leaderboard_stats": leaderboard_stats,
        "categories": categories,
        "diagnostics": {
            "errors": errors[:100],
            "inventory": {},
        },
    }


