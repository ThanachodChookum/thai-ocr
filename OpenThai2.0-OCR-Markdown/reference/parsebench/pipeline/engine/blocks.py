# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared block grammar: an invalid block cannot consume the following block."""
import re

# A class terminator ends a block even when its coordinates are malformed.
BODY = r'((?:(?!<class_|<x_[^<>]*><y_[^<>]*>).)*?)'
NUMBER = r'(\d+(?:\.\d+)?)'
BLOCK_RE = re.compile(
    r'<x_' + NUMBER + r'><y_' + NUMBER + r'>' + BODY +
    r'<x_' + NUMBER + r'><y_' + NUMBER + r'><class_([^>]+)>', re.DOTALL)
TEXT_BLOCK_RE = re.compile(
    r'<x_([^>]*)><y_([^>]*)>' + BODY +
    r'<x_([^>]*)><y_([^>]*)><class_([^>]+)>', re.DOTALL)

CLASS_RE = re.compile(r'<class_([^>]*)>')
OPEN_PAIR_RE = re.compile(r'^\s*<x_[^<>]*><y_[^<>]*>')
CLOSE_PAIR_RE = re.compile(r'<x_[^<>]*><y_[^<>]*>\s*$')
PAIR_RE = re.compile(r'<x_[^<>]*><y_[^<>]*>')
COORDINATE_RE = re.compile(r'<[xy]_[^<>]*(?:>|(?=<)|$)')
CONTROL_RE = re.compile(r'</?s>|<pad>|<unk>|<image>|<tbc>|<\|[^>]+\|>|<(?:predict|output)_[^>]*>')


def visible_text(raw):
    """Remove serialization controls, including incomplete coordinate tokens."""
    return CONTROL_RE.sub('', COORDINATE_RE.sub('', raw)).strip()


def text_blocks(raw):
    """Read text independently of whether coordinates are valid numerical values."""
    cursor = 0
    for match in CLASS_RE.finditer(raw or ''):
        segment = raw[cursor:match.start()]
        pairs = list(PAIR_RE.finditer(segment))
        opening = pairs[-2] if len(pairs) >= 2 else None
        closing = pairs[-1] if pairs and not segment[pairs[-1].end():].strip() else None
        # A dangling preceding block must not become the body of a later,
        # completed block (for example after inserting a valid crop result).
        body = (segment[opening.end():closing.start()].strip()
                if opening and closing and opening.end() <= closing.start()
                else visible_text(segment))
        if body:
            yield match.group(1), body
        cursor = match.end()
    raw_tail = (raw or '')[cursor:]
    tail = visible_text(raw_tail)
    # An unfinished serialized block is retained in raw output, not presented as
    # a successfully decoded text/chart/table block.
    if tail and cursor == 0 and not any(token in raw_tail for token in ['<x_', '<y_']):
        yield 'Text', tail


def has_unparsed_content(raw):
    cursor = 0
    for match in BLOCK_RE.finditer(raw or ''):
        if visible_text(raw[cursor:match.start()]) or COORDINATE_RE.search(raw[cursor:match.start()]):
            return True
        cursor = match.end()
    gap=(raw or '')[cursor:]
    return bool(visible_text(gap) or COORDINATE_RE.search(gap))


def insert_preserving_unparsed(raw, direct_blocks, merged_blocks):
    """Insert additions around validated blocks without discarding malformed text."""
    matches = list(BLOCK_RE.finditer(raw))
    positions = {id(block):match for block,match in zip(direct_blocks,matches)}
    pieces, pending, cursor = [], [], 0
    for block in merged_blocks:
        match = positions.get(id(block))
        if match is None:
            pending.append(block.raw)
            continue
        pieces.append(raw[cursor:match.start()])
        if pending:
            pieces.append('\n\n'.join(pending)+'\n\n')
            pending = []
        pieces.append(raw[match.start():match.end()])
        cursor = match.end()
    pieces.append(raw[cursor:])
    if pending:
        pieces.append('\n\n'+'\n\n'.join(pending))
    return ''.join(pieces)


def incomplete_tail(raw):
    matches=list(CLASS_RE.finditer(raw or ''))
    tail=(raw or '')[matches[-1].end() if matches else 0:]
    return bool(tail.strip()) and (bool(matches) or any(token in tail for token in ['<x_', '<y_']))


def canonical_geometry(raw):
    """Remove bad coordinate controls only from closed, unvalidated blocks.

    Unfinished tails remain verbatim in raw output and receive a diagnostic.
    """
    import math
    terminals=list(CLASS_RE.finditer(raw or ''))
    boundary=terminals[-1].end() if terminals else 0
    closed=(raw or '')[:boundary]
    pieces,cursor=[],0
    for match in BLOCK_RE.finditer(closed):
        pieces.append(COORDINATE_RE.sub('',closed[cursor:match.start()]))
        coords=[match[i] for i in [1,2,4,5]]
        pieces.append(match[0] if all(math.isfinite(float(v)) for v in coords) else COORDINATE_RE.sub('',match[0]))
        cursor=match.end()
    pieces.append(COORDINATE_RE.sub('',closed[cursor:]))
    pieces.append((raw or '')[boundary:])
    return ''.join(pieces)
