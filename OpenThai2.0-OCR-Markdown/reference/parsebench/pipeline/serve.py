# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify the release weights and start the reference one-H100 vLLM profile."""
import argparse,hashlib,importlib.metadata,json,os,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--port',type=int,default=8000);p.add_argument('--host',default='127.0.0.1');a=p.parse_args()
    expected=json.loads((Path(__file__).parent/'expected_results.json').read_text())
    h=hashlib.sha256()
    with (a.model/'model.safetensors').open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected['model_sha256']:raise SystemExit('Weights differ from the evaluated release')
    for name,version in [('nemotron-parse','3.5.0.post2'),('vllm','0.26.0'),('transformers','5.14.1')]:
        if importlib.metadata.version(name)!=version:raise SystemExit(f'Expected {name}=={version}')
    env=os.environ.copy();env['VLLM_PLUGINS']='nemotron_parse_35'
    command=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(a.model.resolve()),
        '--served-model-name','nemotron-parse','--trust-remote-code','--dtype','bfloat16','--host',a.host,'--port',str(a.port),
        '--distributed-executor-backend','uni','--tensor-parallel-size','1','--max-model-len','16384',
        '--max-num-seqs','192','--max-num-batched-tokens','16384','--no-enable-prefix-caching',
        '--attention-config','{"flash_attn_version":3}','--limit-mm-per-prompt','{"image":1}',
        '--chat-template',str(a.model.resolve()/'chat_template.jinja')]
    os.execvpe(sys.executable,command,env)
if __name__=='__main__':main()
