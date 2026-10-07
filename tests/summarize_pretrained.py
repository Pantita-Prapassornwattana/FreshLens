"""Derive candidate/final summaries from recorded predictions, without inference."""
import json
from pathlib import Path
from tests.evaluate_pretrained import metrics

ROOT = Path(__file__).resolve().parents[1]
FILES = ['world_small_validation.json', 'world_medium_validation.json',
         'yoloe_small_validation.json', 'domain_validation.json']


def summarize_classes(report, names):
    rows = [report['per_class'][name] for name in names]
    totals = metrics(sum(row['tp'] for row in rows), sum(row['fp'] for row in rows),
                     sum(row['fn'] for row in rows))
    return {'class_count': len(names), 'matched_class_count': sum(row['tp'] > 0 for row in rows),
            'micro': totals,
            'macro': {key: sum(row[key] if row[key] is not None else 0.0 for row in rows) / len(rows)
                      for key in ['precision', 'recall', 'f1']},
            'macro_zero_division': 0,
            'classes_with_no_true_positive': [name for name in names if report['per_class'][name]['tp'] == 0]}


def overview(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if not report.get('success'):
        raise ValueError(f'Incomplete/failed evaluation: {path}')
    names = list(report['per_class'])
    supported = report['model'].get('supported_classes', names)
    latencies = sorted(item['latency_ms'] for item in report['images'])
    return {'artifact': str(path.relative_to(ROOT)).replace('\\', '/'),
            'model': report['model'], 'split': report['split'],
            'images': report['evaluated_images'], 'confidence_threshold': report['confidence_threshold'],
            'iou_threshold': report['iou_threshold'], 'aggregate': report['aggregate'],
            'all_classes': summarize_classes(report, names),
            'supported_classes': summarize_classes(report, supported),
            'latency': {'mean_ms': sum(latencies) / len(latencies),
                        'median_ms': latencies[len(latencies) // 2],
                        'p95_ms': latencies[int((len(latencies) - 1) * 0.95)]},
            'negative_controls': report['negative_controls']}


def main():
    candidates = [overview(ROOT / 'artifacts' / name) for name in FILES]
    final = overview(ROOT / 'artifacts/pretrained_evaluation.json')
    selected = max(candidates, key=lambda row: row['aggregate']['f1'])
    if final['model']['weights_sha256'] != selected['model']['weights_sha256']:
        raise ValueError('Final test weights differ from validation-selected model')
    output = {'selection_rule': 'Maximum micro F1 across all 47 canonical dataset classes on the same fixed validation sample.',
              'candidates': candidates, 'selected_validation': selected, 'final_test': final,
              'training_runs': 0, 'local_training_epochs': 0,
              'limitations': ['Only sampled subsets were evaluated, not the entire valid/test splits.',
                              'Macro metrics use zero_division=0 and average every named class equally.',
                              'Supported-class subset excludes five unsupported labels and must not replace the all-class score.',
                              'No confidence/prompt/weight changes were made after the final test.',
                              'True external-scene/pretraining independence is unverified.']}
    path = ROOT / 'artifacts/pretrained_comparison.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'path': str(path), 'selected_run_id': selected['model']['run_id'],
                      'final_all_classes': final['all_classes'],
                      'final_supported_classes': final['supported_classes'],
                      'final_latency': final['latency']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
