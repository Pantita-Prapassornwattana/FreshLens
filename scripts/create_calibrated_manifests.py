"""Freeze new accept/reject samples after the earlier expansion trial.

Metadata and hashes only: no model inference, prediction-based sampling, visual
inspection, threshold selection, or changes to the failed trial's records.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.create_expansion_manifests import load_prior, make_manifest, save
from tests.evaluate_pretrained import read_dataset


def summary(path, manifest):
    print(json.dumps({'manifest': str(path.relative_to(ROOT)).replace('\\', '/'),
                      **manifest['summary'], 'class_coverage': manifest['selected_class_coverage'],
                      'unavailable_classes': manifest['classes_unavailable_after_exclusions'],
                      'below_target_classes': manifest['classes_below_per_class_target'],
                      'excluded_images': manifest['excluded_previous_source_images'],
                      'available_images': manifest['available_images_after_exclusions'],
                      'source_group_overlap': manifest['excluded_group_overlap'],
                      'bytes': path.stat().st_size}, ensure_ascii=False), flush=True)


def main():
    artifact = ROOT / 'artifacts'
    old_files = ['pretrained_validation_manifest.json', 'pretrained_evaluation_manifest.json',
                 'expansion_validation_manifest.json', 'expansion_test_manifest.json']
    excluded, references = set(), []
    for name in old_files:
        groups, reference = load_prior(artifact / name)
        excluded.update(groups)
        references.append(reference)
    names, test_rows = read_dataset(ROOT / 'data/raw', 'test')
    test = make_manifest('test', names, test_rows, 3, 20, 313, excluded, references)
    test['purpose'] = 'Fresh final evaluation after frozen validation-only calibration; no tuning on this sample.'
    test['prior_trial_note'] = 'Earlier expansion final test is development evidence and a rejected trial; do not reuse its source groups.'
    path = artifact / 'calibrated_test_manifest.json'
    save(path, test)
    summary(path, test)
    test_groups, test_reference = load_prior(path)
    valid_names, valid_rows = read_dataset(ROOT / 'data/raw', 'valid')
    assert valid_names == names, 'Canonical class names changed'
    confirmation = make_manifest('valid', names, valid_rows, 3, 10, 317,
                                 excluded | test_groups, references + [test_reference])
    confirmation['purpose'] = 'Fresh confirmation validation: frozen configuration accept/reject only, no per-class threshold tuning.'
    confirmation['prior_trial_note'] = test['prior_trial_note']
    confirmation['extra_independence_exclusion'] = 'Also excludes newly frozen calibrated test source groups.'
    valid_path = artifact / 'calibrated_confirmation_manifest.json'
    save(valid_path, confirmation)
    summary(valid_path, confirmation)
    assert not set(test['selected_source_groups']) & set(confirmation['selected_source_groups'])


if __name__ == '__main__':
    main()
