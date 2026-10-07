"""Audit YOLO export and freeze disjoint manifests. Keeps original files untouched."""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent

def digest(path):
    # Pixel hashing detects copies with differing metadata/file encodings.
    with Image.open(path) as im:
        im = im.convert('RGB')
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()

def prepare(source):
    config = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))
    names = config['names']
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    groups, counts, problems, class_counts = {}, Counter(), [], Counter()
    manifests = {}
    content_hashes = {}
    for split, folder in [('test', 'test'), ('val', 'valid'), ('train', 'train')]:
        directory = source / folder / 'images'
        if split == 'val' and not directory.is_dir():
            directory = source / 'val/images'
        files = sorted(p for p in directory.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
        if not files:
            raise ValueError(f'Missing or empty split: {split}')
        kept = []
        for path in files:
            try:
                h = digest(path)
                # Roboflow augmentations typically share the part preceding .rf.
                group = re.split(r'\.rf\.', path.stem)[0]
                group = re.sub(r'_(jpg|jpeg|png)$', '', group)
                keys = ['pixel:' + h, 'source:' + group]
                if any(k in groups and groups[k] != split for k in keys):
                    problems.append({'path': str(path), 'reason': 'cross-split duplicate/source; excluded'})
                    continue
                if 'pixel:' + h in groups:
                    problems.append({'path': str(path), 'reason': 'same-split exact duplicate; excluded'})
                    continue
                label = path.parent.parent / 'labels' / (path.stem + '.txt')
                if not label.is_file():
                    raise ValueError('Missing annotation (use empty txt for negative image)')
                parsed = []
                for line in label.read_text().splitlines():
                    parts = [float(v) for v in line.split()]
                    if len(parts) != 5 or not all(math.isfinite(v) for v in parts):
                        raise ValueError('Invalid YOLO annotation')
                    cls, x, y, w, height = parts
                    if cls != int(cls) or not 0 <= cls < len(names) or not all(0 <= v <= 1 for v in (x,y,w,height)) or w <= 0 or height <= 0:
                        raise ValueError('Annotation outside valid range')
                    parsed.append(int(cls))
                for k in keys:
                    groups[k] = split
                for cls in parsed:
                    class_counts[f'{split}:{names[cls]}'] += 1
                kept.append(str(path.resolve()).replace('\\', '/'))
                content_hashes[str(path.resolve())] = {'pixels': h, 'labels': hashlib.sha256(label.read_bytes()).hexdigest()}
            except Exception as exc:
                problems.append({'path': str(path), 'reason': str(exc)})
        manifests[split] = kept
        counts[split] = len(kept)
    if not all(counts[s] for s in ('train','val','test')):
        raise ValueError('A split is empty after auditing')
    out = ROOT / 'data/prepared'
    out.mkdir(parents=True, exist_ok=True)
    for split, paths in manifests.items():
        (out / f'{split}.txt').write_text('\n'.join(paths)+'\n', encoding='utf-8')
    rng = random.Random(42)
    order = manifests['train'][:]
    rng.shuffle(order)
    for fraction in (0.1, 0.3, 1.0):
        n = max(1, int(len(order)*fraction))
        train = out / f'train_{int(fraction*100)}.txt'
        train.write_text('\n'.join(order[:n])+'\n', encoding='utf-8')
        spec = {'path': str(out.resolve()), 'train': str(train.resolve()), 'val': str((out/'val.txt').resolve()),
                'test': str((out/'test.txt').resolve()), 'names': names}
        (out / f'data_{int(fraction*100)}.yaml').write_text(yaml.safe_dump(spec, allow_unicode=True), encoding='utf-8')
    manifest_hash = hashlib.sha256(json.dumps({'manifests':manifests,'contents':content_hashes}, sort_keys=True).encode()).hexdigest()
    audit = {'source': str(source), 'seed': 42, 'counts': dict(counts), 'names': names,
             'manifest_sha256': manifest_hash, 'class_instances': dict(class_counts), 'excluded': problems, 'content_hashes': content_hashes,
             'limitations': 'Pixel/source-name grouping cannot guarantee near-duplicate or scene-level independence. Review provenance and collect external test images.'}
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/dataset_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts':dict(counts),'excluded':len(problems),'manifest_sha256':manifest_hash}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source', type=Path, help='Extracted YOLO export directory containing data.yaml')
    args=parser.parse_args()
    prepare(args.source.resolve())
