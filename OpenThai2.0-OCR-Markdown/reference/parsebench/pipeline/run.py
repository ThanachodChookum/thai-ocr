# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Run single-, double-, or triple-pass inference against a vLLM endpoint."""
from __future__ import annotations
import argparse,asyncio,base64,hashlib,json,mimetypes,os,tempfile,time,urllib.request
from pathlib import Path
from types import SimpleNamespace
import httpx
from PIL import Image
from engine import geometry
from engine.regions import select_union_regions
from postprocess import process,read_rows,write_rows

PROMPT='</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>'
DETECT='</s><s><predict_bbox><predict_classes><output_no_text><predict_no_text_in_pic>'

def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')

async def requests(items,args,stage):
    semaphore=asyncio.Semaphore(args.concurrency);done=0
    temperature,repetition,cap=(.2,1.05,4096) if stage=='detection' else ((.1,1.,8744) if stage=='crop' else (.2,1.05,8744))
    async with httpx.AsyncClient(timeout=1800,limits=httpx.Limits(max_connections=args.concurrency,max_keepalive_connections=args.concurrency)) as client:
        async def one(item):
            nonlocal done
            async with semaphore:
                path=Path(item['crop_image_path'] if stage=='crop' else item['image_path'])
                encoded=base64.b64encode(path.read_bytes()).decode()
                payload={'model':args.served_model,'messages':[{'role':'user','content':[
                    {'type':'text','text':DETECT if stage=='detection' else PROMPT},
                    {'type':'image_url','image_url':{'url':'data:'+(mimetypes.guess_type(path.name)[0] or 'image/jpeg')+';base64,'+encoded}}]}],
                    'temperature':temperature,'repetition_penalty':repetition,'max_tokens':cap,'top_k':50,'top_p':1.,'seed':20261012,'skip_special_tokens':False}
                start=time.monotonic()
                resp=await client.post(args.endpoint.rstrip('/')+'/chat/completions',json=payload);resp.raise_for_status()
                body=resp.json();choice=body['choices'][0];raw=choice['message'].get('content') or ''
                if not raw.strip():raise RuntimeError('Empty output for '+str(item.get('key') or item.get('crop_id')))
                done+=1
                if done%100==0:print(stage,done,'/',len(items),flush=True)
                return {**item,'raw_out':raw,'usage':body.get('usage',{}),'finish_reason':choice.get('finish_reason'),
                        'error':None,'latency_s':time.monotonic()-start,'coordinate_frame':'content_normalized'}
        return await asyncio.gather(*(one(item) for item in items))

def geometry_args():
    return SimpleNamespace(include_detector_labels='table,chart',detector_order='model',detector_score_min=0.,max_elements_per_page=80,
        fallback_full_page=False,crop_margin_px=12,crop_margin_ratio=.04,min_crop_size=8,model_coordinate_frame='content_normalized',
        canvas_width=1664.,canvas_height=2048.,pad_position='top-left')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images-dir',type=Path,required=True);p.add_argument('--manifest',type=Path)
    p.add_argument('--endpoint',default='http://127.0.0.1:8000/v1');p.add_argument('--served-model',default='nemotron-parse')
    p.add_argument('--mode',choices=['single','double','triple','all'],default='all');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--triple-regions',choices=['detector','union'],default='detector',help='Use separate detector boxes, or add nonoverlapping full-page Table/Chart regions')
    p.add_argument('--concurrency',type=int,default=128);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--keys',help='Comma-separated stable keys for a smoke test')
    p.add_argument('--scratch',type=Path,default=Path(os.environ.get('TMPDIR',tempfile.gettempdir())))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():raise SystemExit('Output already contains a completed run; choose a new directory')
    records=[]
    candidates=read_rows(args.manifest) if args.manifest else [{'image_name':x.name,'key':x.stem.removeprefix('parsebench__')} for x in sorted(args.images_dir.iterdir()) if x.suffix.lower() in ['.jpg','.jpeg','.png','.webp','.tif','.tiff']]
    wanted=set(args.keys.split(',')) if args.keys else None
    for item in candidates:
        k=item.get('key') or Path(item['image_name']).stem.removeprefix('parsebench__')
        if wanted is not None and k not in wanted:continue
        image=args.images_dir/item['image_name']
        with Image.open(image) as im:w,h=im.size
        records.append({'key':k,'image_name':item['image_name'],'image_path':str(image.resolve()),'width':w,'height':h})
    if args.limit:records=records[:args.limit]
    if not records or len({r['key'] for r in records})!=len(records):raise ValueError('No inputs or duplicate keys')
    if wanted is not None and {r['key'] for r in records}!=wanted:raise ValueError('Requested keys missing')
    models=json.loads(urllib.request.urlopen(args.endpoint.rstrip('/')+'/models',timeout=10).read())
    if args.served_model not in {x['id'] for x in models['data']}:raise ValueError('Requested model is not served')
    dump(args.output/'contract.json',{'modes':args.mode,'triple_regions':args.triple_regions,'postprocessing':{'header_inference':'predicted spans and existing bold markup','english_vocabulary':False,'added_header_bold':False,'chart_pair_columns':['Value','Percent'],'literal_duplicate_cleanup':True,'soft_hyphen_cleanup':True,'empty_prefix_repair':True,'caption_association':'plain above-chart only','native_chart_subtitles':True},'pages':len(records),'sampling':{'seed':20261012,'top_k':50,'top_p':1.,'fullpage':[.2,1.05,8744],'detection':[.2,1.05,4096],'crop':[.1,1.,8744]},'coordinate_frame':'content_normalized','crop_margin_px':12,'crop_margin_ratio':.04,'server_models':models,'reads_benchmark_answers':False})
    write_rows(args.output/'inputs.jsonl',records)
    start=time.monotonic();direct=asyncio.run(requests(records,args,'fullpage'));full_seconds=time.monotonic()-start
    write_rows(args.output/'direct.jsonl.gz',direct)
    pages=[geometry.PageImage(key=r['key'],image_name=r['image_name'],image_path=Path(r['image_path']),width=r['width'],height=r['height']) for r in records]
    modes=['single','double','triple'] if args.mode=='all' else [args.mode]
    receipts={}
    for mode in modes:
        mode_start=time.monotonic();folder=args.output/mode;folder.mkdir(exist_ok=True)
        crop_path=None;detections_count=0;crop_count=0
        if mode!='single':
            if mode=='triple':
                detection=asyncio.run(requests(records,args,'detection'));detections_count=len(detection)
            else:detection=direct
            boxes=geometry.parse_model_detection_predictions(geometry_args(),pages,detection)
            if mode=='triple' and args.triple_regions=='union':boxes=select_union_regions(boxes,geometry.parse_model_detection_predictions(geometry_args(),pages,direct))
            write_rows(folder/'detections.jsonl',boxes)
            if mode=='triple':write_rows(folder/'detection_predictions.jsonl.gz',detection)
            with tempfile.TemporaryDirectory(prefix='parse-crops-',dir=args.scratch) as scratch:
                crops=geometry.build_crops(geometry_args(),pages,boxes,Path(scratch)/'images',Path(scratch)/'manifest.jsonl')
                crop_rows=asyncio.run(requests([c.__dict__ for c in crops],args,'crop'))
                crop_count=len(crop_rows);crop_path=folder/'crop_predictions.jsonl';write_rows(crop_path,crop_rows)
        output=process(direct,crop_path,args.workers);write_rows(folder/'predictions.jsonl.gz',output)
        receipts[mode]={'complete':True,'pages':len(output),'passes':{'single':1,'double':2,'triple':3}[mode],
            'fullpage_requests':len(direct),'detection_requests':detections_count,'crop_requests':crop_count,'generation_errors':0,
            'fullpage_seconds':full_seconds,'additional_mode_seconds':time.monotonic()-mode_start,
            'fullpage_shared_across_modes':args.mode=='all','temporary_crops_removed':True}
        dump(folder/'generation_complete.json',receipts[mode]);print(mode,'complete:',len(output),'pages',flush=True)
    dump(args.output/'complete.json',{'complete':True,'modes':receipts,'scoring_performed':False})
if __name__=='__main__':main()
