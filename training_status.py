"""Read recorded training runs and genuine CSV metrics without starting training."""
import csv
import io
import json
import math
from pathlib import Path

FIELDS = {
    'box_train': 'train/box_loss', 'box_val': 'val/box_loss',
    'cls_train': 'train/cls_loss', 'cls_val': 'val/cls_loss',
    'dfl_train': 'train/dfl_loss', 'dfl_val': 'val/dfl_loss',
    'precision': 'metrics/precision(B)', 'recall': 'metrics/recall(B)',
    'map50': 'metrics/mAP50(B)', 'map50_95': 'metrics/mAP50-95(B)',
}

def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

def training_state(root):
    root = Path(root).resolve()
    records = read_json(root / 'artifacts/experiments.json', [])
    records = records if isinstance(records, list) else []
    record = next((r for r in reversed(records) if isinstance(r, dict) and r.get('run_id')), {})
    run_id = str(record.get('run_id', ''))
    csv_path = (root / 'runs' / run_id / 'results.csv').resolve() if run_id else None
    rows = []
    if csv_path and csv_path.is_relative_to(root / 'runs') and csv_path.is_file():
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            for raw in csv.DictReader(stream):
                raw = {k.strip(): v for k, v in raw.items() if k}
                epoch = number(raw.get('epoch'))
                if epoch is None:
                    continue
                row = {'epoch': int(epoch), **{k: number(raw.get(v)) for k, v in FIELDS.items()}}
                if any(row[k] is not None for k in FIELDS):
                    rows.append(row)
    stage = record.get('status', 'not_started')
    if rows and stage == 'not_started':
        stage = 'recorded'
    comparison = read_json(root / 'artifacts/expansion_comparison.json', None)
    return {'status': stage, 'run_id': run_id or None, 'completed_epochs': len(rows),
            'requested_epochs': record.get('requested_epochs'), 'train_images': record.get('train_images'),
            'settings': record.get('settings', {}), 'error': record.get('error'),
            'rows': rows, 'latest': rows[-1] if rows else None,
            'comparison': comparison, 'csv_available': bool(rows)}

def training_csv(state):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['epoch', *FIELDS])
    writer.writeheader()
    writer.writerows(state['rows'])
    return stream.getvalue().encode('utf-8-sig')
