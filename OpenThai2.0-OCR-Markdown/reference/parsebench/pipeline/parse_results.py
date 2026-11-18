# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Parse available model content and isolate failures; no request or time policies."""
from __future__ import annotations

import html
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import postprocess as renderer
from engine import merge
from engine.blocks import canonical_geometry, has_unparsed_content, incomplete_tail, text_blocks


def error(stage, exc):
    return {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}


def fallback(row, errors):
    raw = row.get('raw_out') or ''
    return {'key': renderer.key(row), 'raw_out': raw,
            'source_raw_out': row.get('source_raw_out', raw),
            'markdown': html.escape(raw), 'coordinate_frame': 'content_normalized',
            'postprocessing_status': 'raw_fallback', 'postprocessing_errors': errors}


def finalize(task):
    row, candidates, errors = task
    errors = list(errors)
    original = row.get('raw_out') or ''
    if row.get('error'):
        errors.append({'stage': 'fullpage_generation', 'error': row['error']})
    try:
        raw = renderer.repair_empty_prefix(original)
        if has_unparsed_content(raw) and any(t in raw for t in ['<x_', '<y_', '<class_']):
            errors.append({'stage': 'coordinate_validation',
                           'type': 'IncompleteBlock' if incomplete_tail(raw) else 'MalformedCoordinates'})
        known = renderer.adapter.TEXT_BEARING_CLASSES | {'Table', 'Formula', 'Picture'}
        unknown = sorted({cls for cls, _ in text_blocks(raw) if cls not in known})
        if unknown:
            errors.append({'stage': 'class_validation', 'type': 'UnknownClasses', 'classes': unknown})
        row = dict(row, raw_out=canonical_geometry(raw))
        if row['raw_out'] != original:
            row['source_raw_out'] = original
        if candidates:
            row = merge.build_direct_plus_ordered_structural_addonly(
                renderer.arguments(), [row], {'direct': merge.index_rows([row])},
                {renderer.key(row): candidates})[0]
            # Insertions can place a formerly unfinished tail before a new
            # closed block. Validate that boundary again before rendering.
            row = dict(row, raw_out=canonical_geometry(row['raw_out']), source_raw_out=original)
        result = renderer.finalize_one((row, candidates))
        result['postprocessing_status'] = 'partial' if errors else ('empty' if not original.strip() else 'ok')
        result['postprocessing_errors'] = errors
        if row.get('source_raw_out') is not None:
            result['source_raw_out'] = row['source_raw_out']
        return result
    except Exception as exc:
        return fallback(dict(row, source_raw_out=original), errors + [error('page_rendering', exc)])


def process(direct, crop_path=None, workers=8):
    rows = [dict(row, key=renderer.key(row)) for row in direct]
    if len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input keys')
    args = renderer.arguments()
    classes = merge.parse_class_set(args.structural_classes)
    labels = merge.parse_label_filter(args.include_detector_labels)
    candidates, errors = defaultdict(list), defaultdict(list)
    for crop in renderer.read_rows(crop_path) if crop_path else []:
        key = str(crop.get('page_key') or '')
        if not key:
            continue
        try:
            if crop.get('error'):
                errors[key].append({'stage': 'crop_generation', 'error': crop['error']})
                continue
            if labels and merge.normalize_detector_label(crop.get('detector_label') or '') not in labels:
                continue
            for block in merge.rebuild_crop_row_blocks(crop, args):
                if block.cls in classes and not (args.min_structural_area > 0 and merge.block_area(block) < args.min_structural_area):
                    candidates[key].append(block)
        except Exception as exc:
            errors[key].append(error('crop_reconstruction', exc))
    for blocks in candidates.values():
        blocks.sort(key=merge.reading_order_key)
    tasks = [(row, candidates.get(row['key'], []), errors.get(row['key'], [])) for row in rows]
    if workers == 1:
        return [finalize(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
        return list(pool.map(finalize, tasks, chunksize=8))
