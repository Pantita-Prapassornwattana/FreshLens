"""Download an existing public checkpoint, with revision/hash provenance. No training."""
import hashlib
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT=Path(__file__).resolve().parent
REPO='Piyu12/fruit-veg-yolo11m-detector'
REVISION='82c9865261175e778ba5641cf8560b5d10439ec6'
if __name__=='__main__':
    session=requests.Session()
    session.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1)))
    url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/best.pt'
    target=ROOT/'models/produce35_yolo11m.pt'
    partial=target.with_suffix('.download')
    with session.get(url,stream=True,timeout=(20,60)) as response:
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type',''):
            raise ValueError('Checkpoint response was HTML')
        with partial.open('wb') as output:
            for chunk in response.iter_content(1024*1024):
                if chunk:output.write(chunk)
    if partial.stat().st_size<1_000_000:raise ValueError('Checkpoint unexpectedly small')
    partial.replace(target)
    record={'repository':REPO,'revision':REVISION,'url':url,'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'trained_locally':False}
    (ROOT/'artifacts/expansion_model_download.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record),flush=True)
