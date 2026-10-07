"""Verify the evaluation math instead of testing any model's expected outputs."""
import unittest
from tests.evaluate_pretrained import iou, match, metrics


class DetectionMetricTests(unittest.TestCase):
    def test_duplicate_prediction_only_matches_once_and_wrong_class_never_matches(self):
        predictions = [
            {'class': 'apple', 'confidence': 0.9, 'box': [0, 0, 10, 10]},
            {'class': 'apple', 'confidence': 0.8, 'box': [0, 0, 10, 10]},
            {'class': 'orange', 'confidence': 0.95, 'box': [0, 0, 10, 10]},
        ]
        truth = [{'class': 'apple', 'box': [0, 0, 10, 10]}]
        self.assertEqual(match(predictions, truth), [
            {'prediction_index': 0, 'truth_index': 0, 'iou': 1.0}])

    def test_iou_filters_unlocalized_classification(self):
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(match([{'class': 'apple', 'confidence': 0.99, 'box': [5, 0, 15, 10]}],
                               [{'class': 'apple', 'box': [0, 0, 10, 10]}]), [])

    def test_metrics_handle_no_predictions_and_no_positive_labels(self):
        self.assertEqual(metrics(0, 0, 3), {'tp': 0, 'fp': 0, 'fn': 3,
                                          'precision': None, 'recall': 0.0, 'f1': 0.0})
        self.assertEqual(metrics(0, 0, 0)['f1'], None)
        self.assertEqual(metrics(2, 1, 2)['precision'], 2 / 3)
        self.assertEqual(metrics(2, 1, 2)['recall'], 0.5)


if __name__ == '__main__':
    unittest.main()
