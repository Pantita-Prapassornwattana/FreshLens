"""Download author-provided public pretrained detector; no credentials/training."""
from html.parser import HTMLParser
from pathlib import Path
import argparse
import requests

class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action=None
        self.fields={}
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if tag=='form' and attributes.get('id')=='download-form':
            self.action=attributes.get('action')
        if tag=='input' and attributes.get('type')=='hidden':
            self.fields[attributes['name']]=attributes['value']

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--file-id',default='1XALsPwM4850LkmqsXIfmGhPj8s_nhuiK')
    parser.add_argument('--output',type=Path,default=Path('models/fruit_vegetable_yolov8m.pt'))
    args=parser.parse_args()
    print('Downloading author-provided produce checkpoint',flush=True)
    session=requests.Session()
    response=session.get('https://drive.google.com/uc',params={'export':'download','id':args.file_id},stream=True,timeout=(15,60))
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type',''):
        form=DownloadForm()
        form.feed(response.text)
        response.close()
        if form.action!='https://drive.usercontent.google.com/download':
            raise ValueError('Expected the author-provided public download form')
        response=session.get(form.action,params=form.fields,stream=True,timeout=(15,60))
        response.raise_for_status()
    output=args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.download')
    with temporary.open('wb') as file:
        for chunk in response.iter_content(1024*1024):
            if chunk:file.write(chunk)
    if temporary.stat().st_size<1_000_000:
        raise ValueError('Response is not the checkpoint')
    temporary.replace(output)
    print('Saved bytes',output.stat().st_size,flush=True)
