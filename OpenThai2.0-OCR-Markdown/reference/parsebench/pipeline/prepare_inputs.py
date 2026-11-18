# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Extract only images from an existing ParseBench WebDataset export."""
import argparse,json,tarfile
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--images-out',type=Path,required=True);p.add_argument('--manifest-out',type=Path,required=True);a=p.parse_args()
    a.images_out.mkdir(parents=True,exist_ok=True);rows=[];seen=set()
    for tar in sorted((a.dataset_root/'train').glob('*.tar')):
        with tarfile.open(tar) as t:
            for member in t:
                if not member.isfile() or Path(member.name).suffix.lower() not in ['.jpg','.jpeg','.png','.webp']:continue
                name=Path(member.name).name;k=Path(name).stem
                if k in seen:raise ValueError('Duplicate image key: '+k)
                seen.add(k);target=a.images_out/name
                with t.extractfile(member) as source,target.open('wb') as dest:
                    while chunk:=source.read(1024*1024):dest.write(chunk)
                with Image.open(target) as im:w,h=im.size
                rows.append({'key':k,'image_name':name,'width':w,'height':h})
    if len(rows)!=2078:raise ValueError(f'Expected 2078 unique benchmark pages, got {len(rows)}')
    a.manifest_out.parent.mkdir(parents=True,exist_ok=True)
    a.manifest_out.write_text(''.join(json.dumps(x)+'\n' for x in sorted(rows,key=lambda x:x['key'])))
    print('Prepared',len(rows),'images; no benchmark annotation JSON was read.')
if __name__=='__main__':main()
