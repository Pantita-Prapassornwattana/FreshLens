"""Freeze larger validation and unseen-source test samples without inference.

Reads image headers/labels and computes hashes only. Does not inspect pictures,
load models, request the web API, tune settings, or score the new test sample.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.evaluate_pretrained import read_dataset, select_sample, source_group, negative_controls


def load_prior(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    groups = {source_group(row['path']) for row in payload['sample']
              if row['kind'] != 'synthetic_negative'}
    return groups, {'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'annotated_source_groups': len(groups)}


def count_images_by_class(rows, names):
    counts = Counter()
    for row in rows:
        counts.update({item['class'] for item in row['truth']})
    return {name: counts[name] for name in names}


def make_manifest(split, names, full_rows, per_class, mixed_extra, seed, exclusions, references):
    candidates = [row for row in full_rows if source_group(row['path']) not in exclusions]
    annotated = select_sample(names, candidates, per_class, mixed_extra, seed)
    sample = annotated + negative_controls(ROOT / 'tests/fixtures/expansion_negative_controls')
    for row in sample:
        row['image_sha256'] = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        if row.get('label_path'):
            row['label_sha256'] = hashlib.sha256(Path(row['label_path']).read_bytes()).hexdigest()
    selected_groups = {source_group(row['path']) for row in annotated}
    assert not selected_groups & exclusions, 'Excluded source group entered sample'
    totals = Counter(item['class'] for row in full_rows for item in row['truth'])
    selected_counts = Counter(item['class'] for row in annotated for item in row['truth'])
    available_class_images = count_images_by_class(candidates, names)
    chosen_class_images = count_images_by_class(annotated, names)
    return {
        'created_at': datetime.now(timezone.utc).isoformat(), 'seed': seed, 'split': split,
        'split_image_count': len(full_rows), 'class_names': names,
        'full_split_class_instances': {name: totals[name] for name in names},
        'per_class_target_images': per_class, 'mixed_extra': mixed_extra,
        'available_images_after_exclusions': len(candidates),
        'available_class_images_after_exclusions': available_class_images,
        'classes_unavailable_after_exclusions': [name for name in names if available_class_images[name] == 0],
        'classes_below_per_class_target': [name for name in names if available_class_images[name] < per_class],
        'selected_class_images': chosen_class_images,
        'selected_class_instances': {name: selected_counts[name] for name in names},
        'selected_class_coverage': sum(selected_counts[name] > 0 for name in names),
        'selected_source_groups': sorted(selected_groups),
        'excluded_previous_source_groups': sorted(exclusions),
        'excluded_previous_source_images': len(full_rows) - len(candidates),
        'excluded_manifests': references, 'excluded_group_overlap': 0,
        'inference_performed': False, 'predictions_used_for_selection': False,
        'summary': {'images': len(sample), 'annotated_images': len(annotated),
                    'synthetic_negative_images': len(sample) - len(annotated),
                    'ground_truth_objects': sum(selected_counts.values()),
                    'multi_class_images': sum(len({target['class'] for target in row['truth']}) > 1 for row in annotated)},
        'limitations': [
            'Selection uses class annotations, dimensions, filenames and hashes only; no model scores or visual inspection of new test images.',
            'Roboflow source-name groups are conservative heuristics; generic filenames may exclude unrelated images and near-duplicate scenes can remain.',
            'No guarantee that author pretraining excluded these public images or scenes.',
            'Larger validation can reuse earlier validation examples; it is not an independent final test.',
            'Synthetic negative controls repeat the same four content patterns used previously; they are controls rather than new natural negative evidence.',
            'Stratified samples and source exclusions change the class/scene distribution; scores do not represent the full dataset or production usage.',
            'Freeze configuration using validation before scoring this new test sample; never tune on its scores.'],
        'sample': sample}


def save(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    artifact = ROOT / 'artifacts'
    old_validation_groups, old_validation_ref = load_prior(artifact / 'pretrained_validation_manifest.json')
    old_test_groups, old_test_ref = load_prior(artifact / 'pretrained_evaluation_manifest.json')
    names, valid_rows = read_dataset(ROOT / 'data/raw', 'valid')
    validation = make_manifest('valid', names, valid_rows, 5, 20, 109,
                               old_test_groups, [old_test_ref])
    valid_path = artifact / 'expansion_validation_manifest.json'
    save(valid_path, validation)
    new_validation_groups, new_validation_ref = load_prior(valid_path)
    test_names, test_rows = read_dataset(ROOT / 'data/raw', 'test')
    assert test_names == names, 'Dataset class names changed between splits'
    exclusion = old_validation_groups | old_test_groups | new_validation_groups
    test = make_manifest('test', names, test_rows, 3, 20, 211, exclusion,
                         [old_validation_ref, old_test_ref, new_validation_ref])
    test_path = artifact / 'expansion_test_manifest.json'
    save(test_path, test)
    for path, manifest in [(valid_path, validation), (test_path, test)]:
        print(json.dumps({'manifest': str(path.relative_to(ROOT)).replace('\\', '/'),
                          **manifest['summary'], 'class_coverage': manifest['selected_class_coverage'],
                          'unavailable_classes': manifest['classes_unavailable_after_exclusions'],
                          'below_target_classes': manifest['classes_below_per_class_target'],
                          'excluded_images': manifest['excluded_previous_source_images'],
                          'available_images': manifest['available_images_after_exclusions'],
                          'source_group_overlap': manifest['excluded_group_overlap']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
