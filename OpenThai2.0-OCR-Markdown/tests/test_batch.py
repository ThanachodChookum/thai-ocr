# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import gzip
import json
from pathlib import Path
from io import BytesIO
import base64
import httpx
from PIL import Image
import pytest
from nemotron_parse import PostprocessOptions
from nemotron_parse.inference import DocumentParser, GenerationError
from test_postprocessing import block, tabular


def test_rules_are_enabled_by_default():
    options = PostprocessOptions()
    assert options.split_chart_value_percent and options.relocate_chart_captions


def test_prepared_batch_preserves_input_order_and_pixels(tmp_path):
    paths = []
    for width in [20, 30]:
        p = tmp_path / f"{width}.jpg"
        Image.new("RGB", (width, 20), "red").save(p)
        paths.append(p)
    calls = []

    def backend(image, *, stage, **kwargs):
        calls.append((stage, image.size))
        return block(str(image.width))

    with DocumentParser(backend=backend) as parser:
        prepared = parser.prepare(paths)
        assert prepared.pages[0].data_url.startswith("data:image/jpeg;")
        result = parser.parse_prepared(prepared, postprocess_workers=1)
        assert [d["markdown"] for d in result.documents] == ["20", "30"]
        assert result.metrics["complete_pages"] == 2
        assert result.metrics["postprocessing"]["split_chart_value_percent"] is True
        assert result.metrics["processed_pages_per_second"] > 0
        assert result.metrics["requests"] == {"fullpage": 2, "detection": 0, "crop": 0}


def test_stage_wide_triple_matches_sequential():
    table = tabular(r"\thead{A} & \thead{B}", "a & 10")

    def backend(image, *, stage, **kwargs):
        if stage == "fullpage":
            return block("Before", box=(0, 0, 1, 0.05))
        if stage == "detection":
            return block("", "Table", (0.2, 0.3, 0.8, 0.7))
        return block(table, "Table", (0, 0, 1, 1))

    with DocumentParser(
        backend=backend, crop_margin_px=0, crop_margin_ratio=0
    ) as parser:
        image = Image.new("RGB", (100, 100))
        sequential = parser.parse_triple(image)
        batched = parser.parse_prepared(
            parser.prepare([image]), mode="triple", postprocess_workers=1
        )
    assert batched.documents[0]["markdown"] == sequential["markdown"]
    assert batched.documents[0]["blocks"] == sequential["blocks"]
    assert batched.metrics["requests"] == {"fullpage": 1, "detection": 1, "crop": 1}


def test_preencoded_http_crop_and_fullpage_requests():
    calls = []

    def handle(request):
        body = json.loads(request.content)
        url = body["messages"][0]["content"][1]["image_url"]["url"]
        calls.append(url)
        image = Image.open(BytesIO(base64.b64decode(url.split(",")[1])))
        assert image.mode == "RGB"
        raw = block(tabular("A & 10", "B & 20"), "Table")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": raw}, "finish_reason": "stop"}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with DocumentParser(client=client) as parser:
            result = parser.parse_prepared(
                parser.prepare([Image.new("RGB", (100, 100))]),
                mode="double",
                postprocess_workers=1,
            )
    assert calls[0].startswith("data:image/png;") and calls[1].startswith(
        "data:image/jpeg;"
    )
    assert result.metrics["requests"]["crop"] == 1


def test_partial_outputs_require_explicit_opt_in():
    backend = lambda image, **kwargs: {
        "raw_out": block("partial"),
        "finish_reason": "length",
    }
    with DocumentParser(backend=backend) as parser:
        prepared = parser.prepare([Image.new("RGB", (20, 20))])
        with pytest.raises(GenerationError):
            parser.parse_prepared(prepared, postprocess_workers=1)
        result = parser.parse_prepared(
            prepared, postprocess_workers=1, allow_partial=True
        )
    assert (
        result.metrics["complete_pages"] == 0
        and result.metrics["incomplete_pages"] == 1
    )
    assert result.metrics["token_limited_responses"] == {"fullpage": 1}
    assert result.documents[0]["complete"] is False
    assert result.documents[0]["generation_warnings"]


def test_capped_crop_is_retained_but_not_merged():
    table = tabular("A & old", "B & 20")

    def backend(image, *, stage, **kwargs):
        return {
            "raw_out": block(table, "Table"),
            "finish_reason": "stop" if stage == "fullpage" else "length",
        }

    with DocumentParser(backend=backend) as parser:
        result = parser.parse_prepared(
            parser.prepare([Image.new("RGB", (100, 100))]),
            mode="double",
            allow_partial=True,
            postprocess_workers=1,
        )
    assert result.documents[0]["stages"]["crops"][0]["finish_reason"] == "length"
    assert result.metrics["complete_pages"] == 0
    assert "old" in result.documents[0]["markdown"]


def test_spawned_render_workers_preserve_order():
    with DocumentParser(backend=lambda image, **kw: block(str(image.width))) as parser:
        prepared = parser.prepare([Image.new("RGB", (20 + i, 20)) for i in range(16)])
        result = parser.parse_prepared(prepared, postprocess_workers=2)
    assert [d["markdown"] for d in result.documents] == list(map(str, range(20, 36)))


def test_chart_attributes_are_supported_in_refinement():
    from nemotron_parse.inference import merge_crops

    chart = '<chart type="bar">\ntitle: Sales\n| Item | Value |\n| --- | --- |\n| A | 10 |\n| B | 20 |\n</chart>'
    row = {
        "raw_out": block(chart.replace("10", "11"), "Chart", (0, 0, 1, 1)),
        "detector_label": "chart",
        "detector_score": 1.0,
        "detector_bbox_norm": (0.1, 0.1, 0.9, 0.9),
        "crop_bbox_pixels": (10, 10, 90, 90),
        "page_size": (100, 100),
    }
    assert "| A | 11 |" in merge_crops(block(chart, "Chart"), [row], policy="guarded")


def test_detection_and_crop_overlap_fullpage_with_one_bounded_window():
    import threading

    crop_started = threading.Event()
    lock = threading.Lock()
    active = peak = 0

    def backend(image, *, stage, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if stage == "fullpage":
                assert crop_started.wait(
                    3
                ), "crop should run before full-page parsing finishes"
                return block("Text", box=(0, 0, 1, 0.1))
            if stage == "detection":
                return block("", "Table", (0.2, 0.3, 0.8, 0.7))
            crop_started.set()
            return block(tabular("A & 10", "B & 20"), "Table")
        finally:
            with lock:
                active -= 1

    with DocumentParser(backend=backend) as parser:
        result = parser.parse_prepared(
            parser.prepare([Image.new("RGB", (100, 100))]),
            mode="triple",
            concurrency=2,
            postprocess_workers=1,
        )
    assert peak == result.metrics["peak_inflight_requests"] == 2
    assert result.metrics["requests"] == {"fullpage": 1, "detection": 1, "crop": 1}
    assert result.documents[0]["complete"]


@pytest.mark.parametrize("mode", ["single", "double", "triple"])
@pytest.mark.parametrize("policy", ["union", "detector"])
@pytest.mark.parametrize("merge_policy", ["alternatives", "guarded"])
def test_pipelined_matches_staged_region_selection_and_merge(
    mode, policy, merge_policy
):
    def backend(image, *, stage, **kwargs):
        if stage == "fullpage":
            return block(
                tabular("A & old", "B & 20"), "Table", (0.1, 0.1, 0.45, 0.45)
            ) + block(tabular("C & 30", "D & 40"), "Table", (0.55, 0.55, 0.9, 0.9))
        if stage == "detection":
            return block("", "Table", (0.1, 0.1, 0.45, 0.45))
        return block(tabular("A & new", "B & 20"), "Table")

    with DocumentParser(backend=backend, merge_policy=merge_policy) as parser:
        prepared = parser.prepare([Image.new("RGB", (100, 100)) for _ in range(3)])
        runs = [
            parser.parse_prepared(
                prepared,
                mode=mode,
                triple_regions=policy,
                schedule=s,
                concurrency=2,
                postprocess_workers=1,
            )
            for s in ["staged", "pipelined"]
        ]
    assert runs[0].metrics["requests"] == runs[1].metrics["requests"]
    for a, b in zip(runs[0].documents, runs[1].documents):
        assert a["markdown"] == b["markdown"]
        assert a["blocks"] == b["blocks"]
        assert a["inference"] == b["inference"]
        for stage in ["fullpage", "detection", "crops"]:

            def clean(value):
                if isinstance(value, list):
                    return [clean(v) for v in value]
                return {k: v for k, v in value.items() if k != "latency_s"}

            if stage in a["stages"]:
                assert clean(a["stages"][stage]) == clean(b["stages"][stage])


def test_parse_many_default_can_feed_more_than_four_http_requests():
    import threading

    five_started = threading.Event()
    lock = threading.Lock()
    started = 0

    def handle(request):
        nonlocal started
        with lock:
            started += 1
            if started == 5:
                five_started.set()
        assert five_started.wait(
            5
        ), "HTTP batch default must admit more than four requests"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": block("Text")}, "finish_reason": "stop"}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with DocumentParser(client=client) as parser:
            documents = parser.parse_many(
                [Image.new("RGB", (20, 20)) for _ in range(5)]
            )
    assert len(documents) == started == 5


@pytest.mark.parametrize("schedule", ["staged", "pipelined"])
def test_inference_clock_counts_crop_prep_and_excludes_final_render(
    monkeypatch, schedule
):
    from types import SimpleNamespace
    from nemotron_parse import batch

    # One page makes the dependency chain deterministic without wall-clock sleeps.
    clock = [1000.0]
    monkeypatch.setattr(batch, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original_prep, original_finish = batch._crop_preparer, batch._finish

    def prepare(*args):
        run = original_prep(*args)

        def delayed(*args):
            clock[0] += 7.0
            return run(*args)

        return delayed

    def finish(task):
        clock[0] += 100.0
        return original_finish(task)

    def backend(image, *, stage, **kwargs):
        clock[0] += 3.0
        return block(tabular("A & 10", "B & 20"), "Table")

    monkeypatch.setattr(batch, "_crop_preparer", prepare)
    monkeypatch.setattr(batch, "_finish", finish)
    with DocumentParser(backend=backend) as parser:
        prepared = parser.prepare([Image.new("RGB", (100, 100))])
        result = parser.parse_prepared(
            prepared,
            mode="double",
            concurrency=1,
            postprocess_workers=1,
            schedule=schedule,
        )
    assert result.metrics["inference_seconds"] == 13.0  # 3 + 7 + 3
    assert result.metrics["pipeline_seconds"] == 113.0
    assert result.metrics["inference_pages_per_second"] == pytest.approx(1 / 13)
    assert result.metrics["complete_inference_pages_per_second"] == pytest.approx(
        1 / 13
    )


def test_inference_clock_uses_wall_span_not_sum_of_overlapping_stages():
    from nemotron_parse.batch import _inference_timing

    assert (
        _inference_timing(
            {"fullpage": (10, 20), "detection": (11, 19), "crop": (15, 23)}
        )["inference_seconds"]
        == 13
    )
