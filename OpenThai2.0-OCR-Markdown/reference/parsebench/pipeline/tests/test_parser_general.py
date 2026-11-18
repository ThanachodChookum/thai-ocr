# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocess
from engine import merge, policies
import parse_results


class ParserFixes(unittest.TestCase):
    def test_observed_malformed_footer_retains_text_without_invented_box(self):
        path = Path(__file__).with_name('malformed_coordinate_regression.json')
        for row in json.loads(path.read_text()):
            result = postprocess.process([row], workers=1)[0]
            self.assertEqual(result['source_raw_out'], row['raw_out'])
            self.assertEqual(result['postprocessing_status'], 'partial')
            self.assertIn('Footer text', result['markdown'])
            self.assertEqual(len(merge.parse_blocks(result['raw_out'])), 15)

    def test_malformed_block_does_not_shift_table_replacement(self):
        malformed = '<x_0.1><y_0.1>footer text<x_0.org><y_0.2><class_Page-footer>'
        valid = merge.make_block((.1, .2, .8, .9), 'Table', 'old table')
        candidate = merge.parse_blocks(merge.make_block((.1, .2, .8, .9), 'Table', 'replacement table'))[0]
        result = policies.replace_guarded(malformed+valid, [candidate], merge, similarity=0)
        self.assertTrue(result.startswith(malformed))
        self.assertIn('replacement table', result)

    def test_orphan_coordinates_do_not_swallow_next_valid_block(self):
        raw = '<x_0.1><y_0.1>broken prefix' + merge.make_block((.2, .2, .8, .3), 'Title', 'A valid title')
        blocks = merge.parse_blocks(raw)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].body, 'A valid title')

    def test_bad_page_keeps_other_page_and_original_output(self):
        original = postprocess.finalize_one
        def faulty(task):
            if task[0]['key'] == 'bad':
                raise ValueError('invalid table')
            return original(task)
        with patch.object(postprocess, 'finalize_one', side_effect=faulty):
            rows = postprocess.process([{'key': 'good', 'raw_out': 'readable'}, {'key': 'bad', 'raw_out': 'original text'}], workers=1)
        self.assertEqual(rows[0]['markdown'], 'readable')
        self.assertEqual(rows[1]['postprocessing_status'], 'raw_fallback')
        self.assertEqual(rows[1]['source_raw_out'], 'original text')

    def test_bad_crop_preserves_full_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'crops.jsonl'
            p.write_text(json.dumps({'page_key': 'page', 'detector_label': 'table', 'raw_out': 'bad crop', 'detector_bbox_norm': ['invalid']})+'\n')
            row = postprocess.process([{'key': 'page', 'raw_out': 'full page survives'}], p, workers=1)[0]
        self.assertEqual(row['markdown'], 'full page survives')
        self.assertEqual(row['postprocessing_status'], 'partial')

    def test_unfinished_chart_not_claimed_as_complete(self):
        raw = merge.make_block((.1, .1, .8, .2), 'Text', 'finished text') + '\n<x_0.1><y_0.3><chart type="bar">unfinished data'
        row = postprocess.process([{'key': 'page', 'raw_out': raw}], workers=1)[0]
        self.assertEqual(row['postprocessing_status'], 'partial')
        self.assertIn('finished text', row['markdown'])
        self.assertEqual(row['raw_out'], raw)

    def test_unfinished_table_cannot_contaminate_next_completed_table(self):
        unfinished = '<x_0.1><y_0.1>\\begin{tabular}{cc}\nunfinished & data'
        complete = merge.make_block((.2, .3, .8, .9), 'Table', '\\begin{tabular}{cc}\nleft & right\\\\\n\\end{tabular}')
        from engine.blocks import text_blocks
        self.assertEqual(list(text_blocks(unfinished+complete)), [('Table', '\\begin{tabular}{cc}\nleft & right\\\\\n\\end{tabular}')])
        result = postprocess.process([{'key': 'page', 'raw_out': unfinished+complete}], workers=1)[0]
        self.assertNotIn('unfinished', result['markdown'])
        self.assertIn('left', result['markdown'])
        self.assertIn('unfinished', result['source_raw_out'])

    def test_unterminated_trailing_fragment_is_retained_only_in_raw(self):
        raw = merge.make_block((.1, .1, .8, .2), 'Text', 'completed text') + '\nunfinished trailing text'
        result = postprocess.process([{'key': 'page', 'raw_out': raw}], workers=1)[0]
        self.assertEqual(result['markdown'], 'completed text')
        self.assertIn('unfinished trailing text', result['raw_out'])
        self.assertEqual(result['postprocessing_status'], 'partial')

    def test_plain_input_without_serialized_blocks_still_works(self):
        result = postprocess.process([{'key': 'page', 'raw_out': 'ordinary unstructured text'}], workers=1)[0]
        self.assertEqual(result['markdown'], 'ordinary unstructured text')

    def test_unfinished_serialized_table_only_not_reported_as_parsed(self):
        raw = '<x_0.1><y_0.1>\\begin{tabular}{cc}\nunfinished & data'
        result = postprocess.process([{'key': 'page', 'raw_out': raw}], workers=1)[0]
        self.assertEqual(result['markdown'], '')
        self.assertEqual(result['raw_out'], raw)
        self.assertEqual(result['postprocessing_status'], 'partial')


if __name__ == '__main__':
    unittest.main()
