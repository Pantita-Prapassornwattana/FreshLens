"""Local-only detector using verified pretrained checkpoint; no training required."""
import base64
import io
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
import os
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.yolo'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl'))
for directory in (ROOT / '.yolo', ROOT / '.mpl'):
    directory.mkdir(exist_ok=True)
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)
LOCK = threading.Lock()
MODEL = None
MODEL_PATH = None

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def status():
    selection = read_json(ART / 'selected_model.json', {})
    path = ROOT / selection.get('weights', 'models/best.pt')
    ready = path.is_file() and bool(selection.get('run_id'))
    if selection.get('members'):
        ready=ready and all((ROOT/member['weights']).is_file() for member in selection['members'])
    evaluation=read_json(ROOT/selection.get('evaluation_artifact','artifacts/pretrained_evaluation.json'),None)
    if evaluation is not None:
        evaluation={k:v for k,v in evaluation.items() if k!='images'}
    return {'ready': ready, 'model': selection if ready else None,
            'dataset': read_json(ART / 'dataset_audit.json', read_json(ART/'dataset_inventory.json', None)),
            'evaluation': evaluation,
            'comparison': read_json(ART/'expansion_comparison.json',None) if selection.get('members') else None,
            'experiments': read_json(ART / 'experiments.json', []),
            'history': read_json(ART / 'history.json', [])[-50:][::-1]}

class Handler(BaseHTTPRequestHandler):
    def reply(self, code, payload, content_type='application/json; charset=utf-8'):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split('?')[0]
        if route == '/api/status':
            return self.reply(200, status())
        if route in ('/api/training', '/api/training-csv'):
            from training_status import training_state, training_csv
            training = training_state(ROOT)
            if route == '/api/training-csv':
                if not training['csv_available']:
                    return self.reply(404, {'error': 'ยังไม่มีผลการเทรนราย epoch'})
                return self.reply(200, training_csv(training), 'text/csv; charset=utf-8')
            return self.reply(200, training)
        if route == '/api/training-report':
            path = ROOT / 'docs/training_report.md'
            if not path.is_file():
                return self.reply(404, {'error': 'ยังไม่มีรายงานการเทรน'})
            return self.reply(200, path.read_bytes(), 'text/markdown; charset=utf-8')
        if route == '/training.js':
            return self.reply(200, (ROOT / 'web/training.js').read_bytes(), 'text/javascript; charset=utf-8')
        if route == '/api/examples':
            examples=read_json(ART/'examples.json',[])
            return self.reply(200,[{k:v for k,v in row.items() if k!='path'} for row in examples])
        if route == '/api/example-image':
            example_id=parse_qs(urlparse(self.path).query).get('id',[''])[0]
            row=next((r for r in read_json(ART/'examples.json',[]) if r['id']==example_id),None)
            if row:
                path=(ROOT/row['path']).resolve()
                if path.is_relative_to(ROOT/'data/raw') or path.is_relative_to(ROOT/'tests/fixtures'):
                    return self.reply(200,path.read_bytes(),'image/png' if path.suffix.lower()=='.png' else 'image/jpeg')
            return self.reply(404,{'error':'ไม่พบภาพตัวอย่าง'})
        if route == '/api/report':
            body=(ROOT / 'docs/report.md').read_bytes()
            results=ROOT/'docs/results.md'
            if results.exists():body+=b'\n\n'+results.read_bytes()
            return self.reply(200, body, 'text/markdown; charset=utf-8')
        if route == '/':
            return self.reply(200, (ROOT / 'web/index.html').read_bytes(), 'text/html; charset=utf-8')
        return self.reply(404, {'error': 'ไม่พบหน้านี้'})

    def do_POST(self):
        global MODEL, MODEL_PATH
        if self.path != '/api/predict':
            return self.reply(404, {'error': 'ไม่พบ API'})
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://localhost:8000', 'http://127.0.0.1:8000'):
            return self.reply(403, {'error': 'อนุญาตเฉพาะ localhost'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 12 * 1024 * 1024:
                return self.reply(413, {'error': 'ภาพต้องมีขนาดไม่เกิน 12 MB'})
            s = status()
            if not s['ready']:
                return self.reply(503, {'error': 'โมเดล pretrained ยังไม่พร้อม กรุณารัน setup_pretrained.py ก่อน'})
            from PIL import Image, ImageOps
            image = Image.open(io.BytesIO(self.rfile.read(size)))
            if image.width * image.height > 25_000_000:
                return self.reply(413, {'error': 'ภาพมีความละเอียดสูงเกิน 25 ล้านพิกเซล'})
            image = ImageOps.exif_transpose(image).convert('RGB')
            with LOCK:
                config=s['model']
                key=json.dumps(config,sort_keys=True)
                if MODEL_PATH != key:
                    if MODEL is not None:
                        MODEL.close() if hasattr(MODEL,'close') else MODEL.cpu()
                    if config.get('members'):
                        from produce_detector import ProduceDetector
                        MODEL=ProduceDetector(config)
                    else:
                        from ultralytics import YOLO
                        MODEL=YOLO(str(ROOT/config['weights']))
                    MODEL_PATH=key
                start = time.perf_counter()
                if config.get('members'):
                    boxes,ms=MODEL.predict(image)
                    if self.headers.get('X-Evaluation')=='true':
                        boxes=[b for b in boxes if b['class'] in config.get('evaluation_classes',config['classes'])]
                    from PIL import ImageDraw
                    output=image.copy()
                    draw=ImageDraw.Draw(output)
                    for box in boxes:
                        coords=box['box']
                        draw.rectangle(coords,outline='#e95b70',width=max(2,image.width//200))
                        text=f"{box['class']} {box['confidence']:.2f}"
                        x,y=coords[0],max(0,coords[1]-14)
                        bounds=draw.textbbox((x,y),text)
                        draw.rectangle(bounds,fill='#e95b70')
                        draw.text((x,y),text,fill='white')
                else:
                    class_map=config.get('class_map',{})
                    accepted=[int(i) for i,n in MODEL.names.items() if not class_map or n in class_map]
                    result = MODEL.predict(image, conf=config.get('confidence_threshold',0.25),
                                           iou=config.get('iou_threshold',0.45),imgsz=config.get('imgsz',640),
                                           classes=accepted,device=0,verbose=False)[0]
                    ms = (time.perf_counter() - start) * 1000
                    boxes = [{'class': class_map.get(result.names[int(b.cls.item())],result.names[int(b.cls.item())]), 'confidence': round(float(b.conf.item()), 4),
                              'box': [round(float(x), 1) for x in b.xyxy[0].tolist()]} for b in result.boxes]
                    result.names={int(i):class_map.get(n,n) for i,n in result.names.items()}
                    output = Image.fromarray(result.plot()[..., ::-1])
                buffer = io.BytesIO()
                output.save(buffer, format='JPEG')
                record = {'id': uuid.uuid4().hex, 'time': datetime.now(timezone.utc).isoformat(),
                          'run_id': s['model']['run_id'], 'latency_ms': round(ms, 1), 'detections': boxes}
                if self.headers.get('X-Evaluation')!='true':
                    directory = ART / 'predictions'
                    directory.mkdir(exist_ok=True)
                    (directory / (record['id'] + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                    history = read_json(ART / 'history.json', [])
                    history.append(record)
                    temporary=ART/'history.tmp'
                    temporary.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
                    temporary.replace(ART/'history.json')
                record['image'] = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()
            return self.reply(200, record)
        except (ValueError, OSError):
            return self.reply(400, {'error': 'ไม่สามารถอ่านภาพได้ กรุณาใช้ JPG, PNG หรือ WebP'})
        except Exception as exc:
            print('Inference failed:', repr(exc), flush=True)
            return self.reply(500, {'error': 'ตรวจจับไม่สำเร็จ ตรวจสอบ log ของ server และ dependency'})

if __name__ == '__main__':
    print('FreshLens ready: http://localhost:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
