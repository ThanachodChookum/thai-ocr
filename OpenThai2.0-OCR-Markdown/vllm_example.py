# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import argparse
import base64
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


def _guess_mime(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext in ("webp",):
        return "image/webp"
    # default
    return "image/png"


def _b64_image_data_url(path: str) -> str:
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = _guess_mime(path)
    return f"data:{mime};base64,{img_b64}"


def _iter_images(paths: List[str], image_dir: Optional[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        out.append(p)
    if image_dir:
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            out.extend([str(x) for x in sorted(Path(image_dir).glob(ext))])
    # De-dupe, keep order
    seen = set()
    deduped: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


@dataclass(frozen=True)
class _ReqSpec:
    image_path: str
    request_idx: int


def _make_client(base_url: str, concurrency: int) -> OpenAI:
    # openai>=1.x requires an API key; vLLM ignores it by default.
    import httpx

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    max_connections = max(100, concurrency * 2)
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(
            timeout=httpx.Timeout(1800.0),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        ),
    )


def _run_one(
    req: _ReqSpec,
    *,
    client: OpenAI,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    extra_body: Dict[str, Any],
    image_data_url: Optional[str] = None,
) -> Tuple[_ReqSpec, str, float, Dict[str, Any], Optional[str]]:
    img_url = image_data_url or _b64_image_data_url(req.image_path)
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    latency_s = time.perf_counter() - start
    text = resp.choices[0].message.content or ""
    usage = resp.usage.model_dump() if resp.usage is not None else {}
    finish_reason = resp.choices[0].finish_reason
    return req, text, latency_s, usage, finish_reason


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _maybe_annotate(image_path: str, generated_text: str, out_image_path: str) -> None:
    # Optional visualization (similar to example_with_table_processor.py).
    from PIL import Image, ImageDraw  # local import so batching can run without pillow

    from postprocessing import extract_classes_bboxes, postprocess_text, transform_bbox_to_original

    image = Image.open(image_path).convert("RGB")

    classes, bboxes, texts = extract_classes_bboxes(generated_text)
    bboxes = [transform_bbox_to_original(bbox, image.width, image.height) for bbox in bboxes]

    table_format = "HTML"  # latex | HTML | markdown | json | json_hierarchical | csv
    text_format = "markdown"  # markdown | plain
    blank_text_in_figures = False

    _ = [
        postprocess_text(
            text,
            cls=cls,
            table_format=table_format,
            text_format=text_format,
            blank_text_in_figures=blank_text_in_figures,
        )
        for text, cls in zip(texts, classes)
    ]

    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(
            (bbox[0], bbox[1], max(bbox[0], bbox[2]), max(bbox[1], bbox[3])),
            outline="red",
            width=2,
        )

    image.save(out_image_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="vLLM OpenAI-compatible inference and compact throughput benchmark."
    )
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument(
        "--model",
        default="nemotron-parse",
        help="Served model name (must match vllm serve --served-model-name).",
    )
    ap.add_argument("--image", action="append", default=[], help="Image path (repeatable).")
    ap.add_argument("--image-dir", default=None, help="Directory of images to run (png/jpg/jpeg/webp).")
    ap.add_argument("--out-dir", default="vllm_outputs", help="Where to write .txt outputs.")
    ap.add_argument("--concurrency", type=int, default=4, help="How many concurrent requests to send.")
    ap.add_argument("--max-tokens", type=int, default=8744)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--prompt",
        default="</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
    )
    ap.add_argument("--warmup-count", type=int, default=0, help="Untimed requests before measurement.")
    ap.add_argument(
        "--preload-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Base64-encode images before the timed region.",
    )
    ap.add_argument(
        "--write-text-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write one text file per page (disable for compact benchmark output).",
    )
    ap.add_argument(
        "--predictions-jsonl",
        default=None,
        help="Optional compact scorer-compatible prediction JSONL path.",
    )
    ap.add_argument("--metrics-out", default=None, help="Optional benchmark metrics JSON path.")
    ap.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write annotated images with boxes to --out-dir (default: enabled). Use --no-annotate to disable.",
    )

    args = ap.parse_args()

    image_paths = _iter_images(args.image, args.image_dir)
    if not image_paths:
        raise SystemExit("No images provided. Use --image PATH (repeatable) or --image-dir DIR.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = args.prompt

    extra_body = {
        "repetition_penalty": 1.1,
        "top_k": 1,
        "skip_special_tokens": False,
    }

    reqs: List[_ReqSpec] = []
    for idx, img in enumerate(image_paths):
        reqs.append(_ReqSpec(image_path=img, request_idx=idx))

    prepared_images = (
        {req.image_path: _b64_image_data_url(req.image_path) for req in reqs}
        if args.preload_images
        else {}
    )
    client = _make_client(args.base_url, args.concurrency)

    warmup_reqs = reqs[: min(max(0, args.warmup_count), len(reqs))]
    if warmup_reqs:
        with ThreadPoolExecutor(max_workers=max(1, min(args.concurrency, len(warmup_reqs)))) as ex:
            futures = [
                ex.submit(
                    _run_one,
                    req,
                    client=client,
                    model=args.model,
                    prompt_text=prompt_text,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    extra_body=extra_body,
                    image_data_url=prepared_images.get(req.image_path),
                )
                for req in warmup_reqs
            ]
            for future in as_completed(futures):
                future.result()

    # Concurrency is the simplest way to make sure vLLM batches requests internally.
    summary_lines: List[str] = []
    prediction_rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    completion_tokens = 0
    timed_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [
            ex.submit(
                _run_one,
                r,
                client=client,
                model=args.model,
                prompt_text=prompt_text,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                extra_body=extra_body,
                image_data_url=prepared_images.get(r.image_path),
            )
            for r in reqs
        ]
        for fut in as_completed(futs):
            req, generated_text, latency_s, usage, finish_reason = fut.result()
            base = Path(req.image_path).name
            stem = f"{req.request_idx:04d}_{base}"
            if args.write_text_files:
                out_txt = out_dir / f"{stem}.txt"
                out_txt.write_text(generated_text, encoding="utf-8")
                summary_lines.append(f"{req.image_path}\t{out_txt}")

            latencies.append(latency_s)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            prediction_rows.append(
                {
                    "image_path": req.image_path,
                    "image_name": base,
                    "latency_s": latency_s,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "coordinate_frame": "content_normalized",
                    "raw_out": generated_text,
                    "error": None,
                }
            )

            if args.annotate:
                out_img = out_dir / f"{stem}.annotated.jpg"
                _maybe_annotate(req.image_path, generated_text, str(out_img))

    request_wall_s = time.perf_counter() - timed_start

    if summary_lines:
        (out_dir / "summary.txt").write_text("\n".join(sorted(summary_lines)) + "\n", encoding="utf-8")

    prediction_rows.sort(key=lambda row: str(row["image_name"]))
    if args.predictions_jsonl:
        predictions_path = Path(args.predictions_jsonl)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
            encoding="utf-8",
        )

    metrics = {
        "model": args.model,
        "num_images": len(reqs),
        "num_success": len(prediction_rows),
        "num_failed": len(reqs) - len(prediction_rows),
        "concurrency": args.concurrency,
        "warmup_count": len(warmup_reqs),
        "preload_images": args.preload_images,
        "max_tokens": args.max_tokens,
        "request_wall_s": request_wall_s,
        "pages_per_second": len(prediction_rows) / request_wall_s,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": completion_tokens / request_wall_s,
        "latency_s": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
    }
    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
