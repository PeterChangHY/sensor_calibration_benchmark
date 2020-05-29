#!/usr/bin/env python
""" unit tests for binary_classification_evaluation"""
from unittest2 import TestCase
import os
import sys
import pandas as pd

# add system path for local packages and moudles
try:
    path = os.path.normpath(os.path.join(os.path.abspath(__file__), '../../'))
    sys.path.insert(0, path)
except ValueError:
    print('Can not inser the path {}'.format(path))


from reporter import binary_classification_evaluation
from reporter import data_creator
from reporter import data_plotter


class TestRunner(TestCase):
    """ unit tests """

    def test_binary_classification_evaluation_result(self):
        """unit test for  BinaryClassificationEvaluationResult """
        true_positve = 10
        true_negative = 10
        false_positive = 30
        false_negative = 30

        true_positive_rate = true_positve / float(true_positve + false_negative)
        true_negative_rate = true_negative / float(true_negative + false_positive)
        positive_predictive_value = true_positve / float(true_positve + false_positive)
        negative_predictive_value = true_negative / float(true_negative + false_negative)
        false_negative_rate = false_negative / float(false_negative + true_positve)
        false_positive_rate = false_positive / float(false_positive + true_negative)

        result = binary_classification_evaluation.BinaryClassificationEvaluationResult(
            true_positve, true_negative, false_positive, false_negative)

        self.assertEqual(true_positve, result.true_positive)
        self.assertEqual(true_negative, result.true_negative)
        self.assertEqual(false_positive, result.false_positive)
        self.assertEqual(false_negative, result.false_negative)

        self.assertAlmostEqual(true_positive_rate, result.true_positive_rate, delta=0.001)
        self.assertAlmostEqual(true_negative_rate, result.true_negative_rate, delta=0.001)
        self.assertAlmostEqual(positive_predictive_value,
                               result.positive_predictive_value, delta=0.001)
        self.assertAlmostEqual(negative_predictive_value,
                               result.negative_predictive_value, delta=0.001)
        self.assertAlmostEqual(false_negative_rate, result.false_negative_rate, delta=0.001)
        self.assertAlmostEqual(false_positive_rate, result.false_positive_rate, delta=0.001)

    def test_binary_classification_evaluation(self):
        outcome_data = data_creator.Data()
        outcome_data.key = 'Test'
        outcome_data.value_type = 'boolean'
        tmp_data_list = []
        outcome_start_time = 1578784.3017
        for t in range(0, 10):
            tmp_data_list.append([outcome_start_time + t, False])
        for t in range(10, 20):
            tmp_data_list.append([outcome_start_time + t, True])

        outcome_data.data_frame = pd.DataFrame(tmp_data_list, columns=['Unix_Time', 'Parsed_value'])

        ground_truth_data = data_creator.Data()
        ground_truth_data.key = 'Test'
        ground_truth_data.value_type = 'boolean'
        tmp_data_list = []
        ground_truth_start_time = 1578784.3417
        for t in range(0, 10):
            tmp_data_list.append([ground_truth_start_time + t, True])
        for t in range(10, 20):
            tmp_data_list.append([ground_truth_start_time + t, False])

        ground_truth_data.data_frame = pd.DataFrame(tmp_data_list, columns=['Unix_Time', 'Parsed_value'])

        evaluation = binary_classification_evaluation.BinaryClassificationEvaluation(outcome_data,
                                                                                    ground_truth_data,
                                                                                    match_time_tolerance_in_second=0.05,
                                                                                    default_value=False)

        self.assertEqual(0, evaluation.evaluation_result.true_positive)
        self.assertEqual(0, evaluation.evaluation_result.true_negative)
        self.assertEqual(10, evaluation.evaluation_result.false_positive)
        self.assertEqual(10, evaluation.evaluation_result.false_negative)
        self.assertEqual(0, evaluation.evaluation_result.true_negative_rate)
        self.assertEqual(0, evaluation.evaluation_result.true_negative_rate)
        self.assertEqual(0, evaluation.evaluation_result.positive_predictive_value)
        self.assertEqual(0, evaluation.evaluation_result.negative_predictive_value)
        self.assertEqual(1.0, evaluation.evaluation_result.false_positive_rate)
        self.assertEqual(1.0, evaluation.evaluation_result.false_negative_rate)

        data_plotter.generate_binary_plot(outcome_data, ground_truth_data, 'test_plot', '/home/peterchang/work/test.png')
