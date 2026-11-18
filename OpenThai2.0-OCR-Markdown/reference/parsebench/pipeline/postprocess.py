# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prediction-only merge and Markdown conversion; no labels or scorer imports."""
from __future__ import annotations
import argparse,gzip,json,re,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from engine import merge,markdown as adapter,policies
from engine.header_inference_probe import infer_header_rows
from engine.text_cleanup import normalize_text,repair_empty_prefix,dedupe_exact_elements
from engine.caption_relocation import relocate_above_captions
from engine.chart_context import add_native_subtitles
from engine.chart_caption import transform

def read_rows(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:
        return [json.loads(s) for s in f if s.strip()]

def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(path,'wt') if str(path).endswith('.gz') else path.open('w')) as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')

def key(row):return Path(row.get('key') or row.get('image_name','')).stem.removeprefix('parsebench__')

def arguments(crops=None):
    return SimpleNamespace(crop_element_predictions=[Path(crops)] if crops else [],structural_classes='Table,Chart',
        include_detector_labels='chart,table',max_crop_output_chars=0,max_crop_output_blocks=0,min_structural_area=0.,
        crop_coordinate_frame='content_normalized',crop_pad_position='top-left',prediction_pad_position='top-left',
        output_coordinate_frame='content_normalized',canvas_width=1664.,canvas_height=2048.,structural_source='crop',
        dedupe_overlap_threshold=.85,dedupe_text_threshold=.8,replacement_text_threshold=.8)

def corrected(raw):
    text=adapter.mmd_to_markdown(raw,text_transform=normalize_text)
    return re.sub(r'<table\b[^>]*>.*?</table>',lambda m:infer_header_rows(m[0])[0],text,flags=re.S|re.I)

def _markdown_base(raw):
    raw=policies.chart_render(raw,panels=True,compound=True)
    def title(m):
        if m[4]!='Chart':return m[0]
        html,n=transform(corrected(m[0]))
        return m[1]+html+m[3] if n else m[0]
    return policies.promote_bold_header_row(corrected(policies.BLOCK.sub(title,raw)))

def markdown(raw):
    text=_markdown_base(raw)
    return add_native_subtitles(relocate_above_captions(raw,text,_markdown_base))

def finalize_one(task):
    row,candidates=task;k=key(row)
    raw=policies.replace_guarded(row['raw_out'],candidates,merge,guard='columns',repair=True,emit_repair=False)
    raw=dedupe_exact_elements(repair_empty_prefix(raw))
    return {'key':k,'raw_out':raw,'markdown':markdown(raw),'coordinate_frame':'content_normalized'}

def process(direct,crop_path=None,workers=8):
    from parse_results import process as parse_process
    return parse_process(direct,crop_path,workers)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--direct',type=Path,required=True)
    p.add_argument('--crops',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    start=time.monotonic();rows=process(read_rows(a.direct),a.crops,a.workers);write_rows(a.output,rows)
    print(json.dumps({'pages':len(rows),'seconds':time.monotonic()-start,'output':str(a.output)}))
if __name__=='__main__':main()
