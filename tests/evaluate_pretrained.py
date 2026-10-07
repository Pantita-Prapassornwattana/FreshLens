"""Deterministic held-out API validation; never imports/trains a model.

Select two images per available split class, extra mixed-class scenes, and four
synthetic negative controls. Saves the exact inputs, outputs and IoU matches.
These are fixed-threshold detection metrics, NOT COCO mAP or a random-sample
estimate of production accuracy. Run once after freezing the model/settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match(predictions, truth, threshold=0.5):
    """One-to-one, same-class matching in descending confidence order."""
    used = set()
    matches = []
    for index in sorted(range(len(predictions)), key=lambda i: predictions[i]['confidence'], reverse=True):
        prediction = predictions[index]
        candidates = [(iou(prediction['box'], target['box']), j)
                      for j, target in enumerate(truth)
                      if j not in used and prediction['class'] == target['class']]
        best = max(candidates, default=(0.0, -1))
        if best[0] >= threshold and best[1] >= 0:
            used.add(best[1])
            matches.append({'prediction_index': index, 'truth_index': best[1], 'iou': round(best[0], 6)})
    return matches


def read_dataset(source, split='valid'):
    spec = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))
    names = spec['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    rows = []
    for image in sorted((source / split / 'images').iterdir()):
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
            continue
        label = source / split / 'labels' / f'{image.stem}.txt'
        if not label.is_file():
            raise ValueError(f'Missing {split} label: {label}')
        truth = []
        with Image.open(image) as picture:
            width, height = picture.size
        for line in label.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            numbers = [float(v) for v in line.split()]
            if len(numbers) != 5 or not all(math.isfinite(v) for v in numbers):
                raise ValueError(f'Invalid annotation: {label}')
            cls, x, y, w, h = numbers
            if cls != int(cls) or not 0 <= cls < len(names) or not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f'Invalid annotation coordinates: {label}')
            truth.append({'class': names[int(cls)], 'box': [
                (x - w / 2) * width, (y - h / 2) * height,
                (x + w / 2) * width, (y + h / 2) * height]})
        rows.append({'path': str(image.resolve()), 'label_path': str(label.resolve()),
                     'truth': truth, 'width': width, 'height': height,
                     'kind': 'annotated_' + split})
    return names, rows


def select_sample(names, rows, per_class, mixed_extra, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        for label in {item['class'] for item in row['truth']}:
            buckets[label].append(row)
    chosen = {}
    for name in names:
        candidates = sorted(buckets[name], key=lambda row: row['path'])
        rng.shuffle(candidates)
        for row in candidates[:per_class]:
            chosen[row['path']] = row
    mixed = [row for row in rows if len({target['class'] for target in row['truth']}) > 1
             and row['path'] not in chosen]
    rng.shuffle(mixed)
    for row in mixed[:mixed_extra]:
        chosen[row['path']] = row
    # Include actual annotated negatives if present; this dataset has none.
    negative = [row for row in rows if not row['truth'] and row['path'] not in chosen]
    rng.shuffle(negative)
    for row in negative[:4]:
        chosen[row['path']] = row
    return sorted(chosen.values(), key=lambda row: row['path'])


def source_group(path):
    stem = Path(path).stem.split('.rf.')[0]
    return re.sub(r'_(jpg|jpeg|png)$', '', stem)


def negative_controls(directory):
    directory.mkdir(parents=True, exist_ok=True)
    pictures = {'white': Image.new('RGB', (640, 480), 'white'),
                'black': Image.new('RGB', (640, 480), 'black'),
                'checkerboard': Image.new('RGB', (640, 480), 'white'),
                'geometric_shapes': Image.new('RGB', (640, 480), '#ececec')}
    draw = ImageDraw.Draw(pictures['checkerboard'])
    for y in range(0, 480, 40):
        for x in range(0, 640, 40):
            if (x // 40 + y // 40) % 2:
                draw.rectangle((x, y, x + 39, y + 39), fill='black')
    draw = ImageDraw.Draw(pictures['geometric_shapes'])
    draw.rectangle((40, 50, 230, 230), fill='#2463ad')
    draw.polygon([(380, 40), (520, 250), (280, 250)], fill='#9b59b6')
    draw.line((30, 350, 600, 350), fill='black', width=20)
    rows = []
    for name, picture in pictures.items():
        path = directory / f'{name}.png'
        picture.save(path)
        rows.append({'path': str(path.resolve()), 'truth': [], 'kind': 'synthetic_negative',
                     'width': 640, 'height': 480})
    return rows


def metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def request_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--manifest', type=Path, help='Reuse a previously frozen sample for candidate comparison')
    parser.add_argument('--exclude-manifest', type=Path, help='Exclude previously sampled source groups from a new final test sample')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--per-class', type=int, default=2)
    parser.add_argument('--mixed-extra', type=int, default=10)
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/pretrained_validation.json')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.per_class < 1 or args.mixed_extra < 0 or not 0 <= args.iou <= 1 or not 0 <= args.confidence <= 1:
        parser.error('Invalid sample or threshold setting')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        names, sample = manifest['class_names'], manifest['sample']
        args.split = manifest['split']
        for row in sample:
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError(f"Image changed since manifest was frozen: {row['path']}")
            if row.get('label_path') and hashlib.sha256(Path(row['label_path']).read_bytes()).hexdigest() != row['label_sha256']:
                raise ValueError(f"Annotation changed since manifest was frozen: {row['label_path']}")
    else:
        names, all_rows = read_dataset(args.source.resolve(), args.split)
        candidates = all_rows
        excluded_groups = set()
        if args.exclude_manifest:
            previous = json.loads(args.exclude_manifest.read_text(encoding='utf-8'))
            excluded_groups = {source_group(row['path']) for row in previous['sample']
                               if row['kind'] != 'synthetic_negative'}
            candidates = [row for row in all_rows if source_group(row['path']) not in excluded_groups]
        sample = select_sample(names, candidates, args.per_class, args.mixed_extra, args.seed)
        sample += negative_controls(ROOT / 'tests/fixtures/negative_controls')
        for row in sample:
            row['image_sha256'] = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
            if row.get('label_path'):
                row['label_sha256'] = hashlib.sha256(Path(row['label_path']).read_bytes()).hexdigest()
        totals = Counter(item['class'] for row in all_rows for item in row['truth'])
        manifest = {'seed': args.seed, 'split': args.split, 'split_image_count': len(all_rows), 'class_names': names,
                    'full_split_class_instances': {name: totals[name] for name in names},
                    'excluded_previous_source_groups': sorted(excluded_groups),
                    'excluded_previous_source_images': len(all_rows) - len(candidates),
                    'per_class_target_images': args.per_class, 'mixed_extra': args.mixed_extra,
                    'sample': sample}
        manifest_path = args.output.with_name(args.output.stem + '_manifest.json')
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(manifest_path), 'images': len(sample),
                      'classes': len(names), 'mode': 'plan_only' if args.plan_only else 'evaluate'}), flush=True)
    if args.plan_only:
        return
    status = request_json(args.base_url.rstrip('/') + '/api/status')
    if not status.get('ready'):
        raise RuntimeError('Server reports no ready model. No evaluation was attempted.')
    counts = defaultdict(Counter)
    records, failures = [], []
    started = time.perf_counter()
    for number, row in enumerate(sample, start=1):
        item = dict(row)
        data = Path(row['path']).read_bytes()
        content_type = 'image/png' if Path(row['path']).suffix.lower() == '.png' else 'image/jpeg'
        request = urllib.request.Request(args.base_url.rstrip('/') + '/api/predict', data=data,
            headers={'Content-Type': content_type, 'X-Evaluation': 'true'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                prediction = json.load(response)
            expected_run = (status.get('model') or {}).get('run_id')
            if expected_run and prediction.get('run_id') != expected_run:
                raise ValueError('Server model changed during evaluation; rerun with one frozen candidate')
            detections = [d for d in prediction['detections'] if d['confidence'] >= args.confidence]
            if any(d['class'] not in names for d in detections):
                raise ValueError('Unknown class returned by API; canonical names must match dataset')
            matches = match(detections, row['truth'], args.iou)
            matched_predictions = {v['prediction_index'] for v in matches}
            matched_truth = {v['truth_index'] for v in matches}
            for index, detection in enumerate(detections):
                counts[detection['class']]['tp' if index in matched_predictions else 'fp'] += 1
            for index, truth in enumerate(row['truth']):
                if index not in matched_truth:
                    counts[truth['class']]['fn'] += 1
            item.update({'detections': detections, 'matches': matches,
                         'latency_ms': prediction.get('latency_ms'), 'run_id': prediction.get('run_id'),
                         'tp': len(matches), 'fp': len(detections) - len(matches),
                         'fn': len(row['truth']) - len(matches)})
        except (urllib.error.URLError, ValueError, KeyError) as exc:
            item['error'] = str(exc)
            failures.append({'path': row['path'], 'error': str(exc)})
        records.append(item)
        print(json.dumps({'image': number, 'total': len(sample), 'file': Path(row['path']).name,
                          'tp': item.get('tp'), 'fp': item.get('fp'), 'fn': item.get('fn'),
                          'error': item.get('error')}), flush=True)
        # Preserve partial results if an inference call or process later fails.
        report = {'created_at': datetime.now(timezone.utc).isoformat(), 'complete': number == len(sample),
                  'success': not failures and number == len(sample), 'model': status.get('model'),
                  'confidence_threshold': args.confidence, 'iou_threshold': args.iou,
                  'split': args.split,
                  'manifest': str(manifest_path), 'elapsed_seconds': round(time.perf_counter() - started, 2),
                  'attempted_images': number, 'planned_images': len(sample),
                  'evaluated_images': number - len(failures), 'failures': failures,
                  'per_class': {name: metrics(counts[name]['tp'], counts[name]['fp'], counts[name]['fn'])
                                for name in names},
                  'aggregate': metrics(sum(c['tp'] for c in counts.values()),
                                       sum(c['fp'] for c in counts.values()),
                                       sum(c['fn'] for c in counts.values())),
                  'negative_controls': {'images': sum(r['kind'] == 'synthetic_negative' and 'error' not in r for r in records),
                                        'false_detections': sum(r.get('fp', 0) for r in records if r['kind'] == 'synthetic_negative')},
                  'limitations': [
                      'Stratified small held-out sample; scores are not a full-dataset or production accuracy estimate.',
                      'Validation can compare candidates; final test should run only after model and settings are frozen.',
                      'Metrics are precision/recall/F1 at fixed confidence and IoU, not COCO mAP.',
                      'Roboflow annotations may be incomplete or ambiguous; mismatched class labels count as FP and FN.',
                      'Pretraining data may overlap this public dataset; true external-scene independence is unverified.',
                      'Previous-source exclusions use Roboflow filename grouping; generic names can overexclude distinct images and near duplicates can remain.',
                      'Synthetic negative controls test empty/geometric inputs only, not natural non-produce scenes.',
                      'No training or fine tuning occurs in this script.'], 'images': records}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'aggregate': report['aggregate'],
                      'negative_controls': report['negative_controls'], 'failures': len(failures)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
