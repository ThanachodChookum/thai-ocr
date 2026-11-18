# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prepare public ParseBench PDFs and annotations as compact evaluation shards."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import tarfile

FILES = ['chart.jsonl', 'layout.jsonl', 'table.jsonl', 'text_content.jsonl', 'text_formatting.jsonl']
SNAPSHOT = None


def load_groups(snapshot):
    groups = {}
    for name in FILES:
        with (snapshot / name).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                relative = row['pdf']
                group = groups.setdefault(relative, {'categories': set(), 'rules': [], 'ground_truth': ''})
                category = row.get('category') or Path(name).stem
                group['categories'].add(category)
                details = row.get('rule') or {}
                if isinstance(details, str):
                    details = json.loads(details)
                rule = {'type': row['type'], **details}
                for field in ['id', 'page', 'tags', 'category']:
                    if row.get(field) is not None and row.get(field) != []:
                        rule[field] = row[field]
                group['rules'].append(rule)
                if row.get('expected_markdown') and not group['ground_truth']:
                    group['ground_truth'] = row['expected_markdown']
    return sorted(groups.items())


def render(task):
    import pymupdf as fitz
    index, relative = task
    relative_path = Path(relative)
    if relative_path.is_absolute() or '..' in relative_path.parts:
        raise ValueError('Source outside dataset directory: ' + relative)
    # Hugging Face snapshot entries are normally symlinks to its blob cache.
    source = SNAPSHOT / relative_path
    with fitz.open(source) as doc:
        if len(doc) < 1:
            raise ValueError('Empty document: ' + relative)
        pix = doc[0].get_pixmap(dpi=150, alpha=False)
        image = pix.tobytes('jpeg', jpg_quality=92)
        return index, relative, image, pix.width, pix.height, len(doc)


def add(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def main():
    global SNAPSHOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    SNAPSHOT = args.snapshot
    groups = load_groups(SNAPSHOT)
    if len(groups) != 2078:
        raise ValueError(f'Expected 2078 unique pages, got {len(groups)}')
    import pymupdf as fitz
    if fitz.VersionBind != '1.26.3':
        raise SystemExit('Use PyMuPDF==1.26.3 for the recorded rendering contract')
    train = args.output / 'train'
    if train.exists() and any(train.iterdir()):
        raise SystemExit('Choose a new output directory; existing shards are preserved')
    train.mkdir(parents=True, exist_ok=True)
    manifest = []
    archive = None
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
            for index, relative, image, width, height, source_pages in pool.map(render, enumerate(name for name, group in groups), chunksize=4):
                if index % 500 == 0:
                    if archive is not None:
                        archive.close()
                    archive = tarfile.open(train / f'part-{index // 500:06d}.tar', 'w')
                key = f'parsebench_{index:05d}'
                group = groups[index][1]
                category = sorted(group['categories'])[0]
                expected = group['ground_truth']
                annotation = {'block_boxes': [[0, 0, width, height]], 'block_classes': ['Text'],
                    'block_text': [expected], 'ground_truth': expected, 'rules': group['rules'],
                    'category': category, 'page_info': {'width': width, 'height': height}}
                add(archive, key + '.jpg', image)
                add(archive, key + '.json', json.dumps(annotation, ensure_ascii=False).encode())
                manifest.append({'key': key, 'source_pdf': relative, 'image_name': key + '.jpg',
                    'width': width, 'height': height, 'source_pages': source_pages, 'source_page_index': 0,
                    'image_sha256': hashlib.sha256(image).hexdigest()})
                if (index + 1) % 200 == 0:
                    print('Prepared', index + 1, '/', len(groups), flush=True)
    finally:
        if archive is not None:
            archive.close()
    (args.output / 'source_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in manifest))
    receipt = {'complete': True, 'pages': len(manifest), 'renderer': 'PyMuPDF', 'renderer_version': fitz.VersionBind,
        'dpi': 150, 'jpeg_quality': 92, 'source_page_index': 0,
        'multi_page_sources': [row['source_pdf'] for row in manifest if row['source_pages'] > 1],
        'annotations_used_only_for_dataset_preparation_and_scoring': True,
        'annotation_files': {name: hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() for name in FILES},
        'shards': {path.name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(train.glob('*.tar'))}}
    (args.output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
