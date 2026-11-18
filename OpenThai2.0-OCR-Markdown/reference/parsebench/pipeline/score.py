# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fresh official scoring, separate from prediction-only inference."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,multiprocessing,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from engine import score_adapter as adapter
from postprocess import read_rows
SCORER=None

def one(row):
    # The renderer already produced the final Markdown. The scorer still receives
    # the unchanged final boxes/classes and the frozen adapter's layout mapping.
    adapter.mmd_to_markdown=lambda raw:row['markdown']
    results=SCORER.score_record({'key':row['key'],'raw_out':row['raw_out'],'coordinate_frame':'content_normalized'})
    return {'key':row['key'],'results':[{k:r.get(k) for k in ['category','success','error','metrics']} for r in results]}

def main():
    global SCORER
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    expected=json.loads((Path(__file__).parent/'scorer_source_hashes.json').read_text())
    for name,sha in expected.items():
        f=a.source/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:raise ValueError('Scorer source mismatch: '+name)
    sys.path.insert(0,str(a.source.resolve()));adapter.ensure_parsebench_imports=lambda *args,**kwargs:None
    SCORER=adapter.OfficialParseBenchScorer(data_root=a.dataset_root)
    rows=read_rows(a.predictions);keys={x['key'] for x in rows}
    if len(keys)!=len(rows) or not rows:raise ValueError('Duplicate keys or empty predictions')
    if not a.allow_partial and len(rows)!=2078:raise ValueError('Full scoring requires 2078 pages')
    a.output.mkdir(parents=True,exist_ok=True);groups=collections.defaultdict(list);errors=[];start=time.monotonic();count=0
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('fork')) as pool,gzip.open(a.output/'per_page_scores.jsonl.gz','wt') as f:
        for row in pool.map(one,rows,chunksize=1):
            count+=1;f.write(json.dumps(row)+'\n')
            for result in row['results']:
                groups[result['category']].append(result)
                if not result['success']:errors.append({'key':row['key'],'result':result})
            if count%100==0:print('Scored',count,'/',len(rows),flush=True)
    aggregate=adapter.aggregate_official_results(groups,len(rows),collections.Counter(),[])
    fields={'table':('grits_trm_composite','tables'),'chart':('rule_chart_data_point_pass_rate','charts'),
            'text_content':('content_faithfulness','content_faithfulness'),'text_formatting':('semantic_formatting','semantic_formatting'),
            'layout':('layout_element_rule_pass_rate','visual_grounding')}
    denominators={};independent={}
    for category,(metric,field) in fields.items():
        vals=[r['metrics'][metric] for r in groups[category] if metric in r['metrics']]
        denominators[field]=len(vals)
        if vals:independent[field]=sum(vals)/len(vals)
    if len(independent)==5:
        independent['overall']=sum(independent.values())/5
        assert all(abs(independent[k]-aggregate['leaderboard'][k])<1e-12 for k in independent)
    receipt={'complete':not errors,'pages':len(rows),'fresh_page_evaluations':count,'reused_page_scores':0,'official_errors':len(errors),
        'scores':aggregate['leaderboard'],'independent_scores':independent,'denominators':denominators,
        'scorer_revision':'9af4102e26b51f234ab5f037e19fec05c0696ec6','verified_source_files':len(expected),
        'prediction_sha256':hashlib.sha256(a.predictions.read_bytes()).hexdigest(),'seconds':time.monotonic()-start,'errors':errors}
    (a.output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    if errors:raise SystemExit('Official scoring errors; see complete.json')
if __name__=='__main__':main()
