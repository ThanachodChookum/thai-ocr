import sys
from pathlib import Path
import httpx
sys.path.insert(0, "./parse35")
from nemotron_parse.inference import DocumentParser

PROFILES = {'single': {'concurrency': 128, 'keepalive': 'pooled'}, 'double': {'concurrency': 128, 'keepalive': 'pooled'}, 'triple': {'concurrency': 256, 'keepalive': 'pooled'}}

def main():
    mode = sys.argv[1]
    profile = PROFILES[mode]
    concurrency = profile["concurrency"]
    keepalive = 0 if profile["keepalive"] == 0 else max(64, concurrency)
    pages = sys.argv[2:]
    with httpx.Client(
        timeout=1800, limits=httpx.Limits(
            max_connections=max(64, concurrency),
            max_keepalive_connections=keepalive, keepalive_expiry=1.0,
        ),
    ) as client:
        with DocumentParser(
            base_url="http://localhost:8000/v1", model="nemotron-parse", client=client,
        ) as parser:
            prepared = parser.prepare(pages, workers=8)
            result = parser.parse_prepared(
                prepared, mode=mode, concurrency=concurrency, postprocess_workers=8,
                triple_regions="union", schedule="pipelined", allow_partial=True,
            )
    print(result.metrics)

if __name__ == "__main__":
    main()
