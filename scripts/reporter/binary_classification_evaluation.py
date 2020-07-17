#!/usr/bin/env python
"""this moudle include a binary classification evaluation """

from match_data_frame import match_time_and_fill

import time


class BinaryClassificationEvaluationResult(object):
    """
        Data class contains:

        basic classcifications:
        1. true positive
        2. true negative
        3. false positive
        4. false negative

        ratios:
        1. true positive rate (TPR), aka sensitivity or recall
        2. true negative rate (TNR), aka specificity
        3. positive predictive value (PPV), aka precision
        4. negative predictive value (NPV)
        5. false negative rate (FNR), aka miss rate
        6. false positive rate (FPR), aka fall-out
        7. false discovery rate (FDR)
        8. false omission rate (FOR)

        others:
        1. total_number: total number of classcifications

    """

    def __init__(self, true_positive=0, true_negative=0, false_positive=0, false_negative=0):
        self.true_positive_rate = 0
        self.true_negative_rate = 0
        self.positive_predictive_value = 0
        self.negative_predictive_value = 0
        self.false_positive_rate = 0
        self.false_negative_rate = 0
        self.false_discovery_rate = 0
        self.false_omission_rate = 0
        self.f1_score = 0
        self.total_number = 0

        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

        self._cal_ratios()
        self._cal_total_number()

    def add(self, other):
        """ combine other result"""
        self.true_positive = self.true_positive + other.true_positive
        self.true_negative = self.true_negative + other.true_negative
        self.false_positive = self.false_positive + other.false_positive
        self.false_negative = self.false_negative + other.false_negative
        self._cal_ratios()
        self._cal_total_number()

    def _cal_ratios(self):
        self._gen_true_positive_rate()
        self._gen_true_negative_rate()
        self._gen_positive_predictive_value()
        self._gen_negative_predictive_value()
        self._gen_false_positive_rate()
        self._get_false_negative_rate()
        self._get_false_discovery_rate()
        self._get_false_omission_rate()

        self._get_f1_score()

    def _cal_total_number(self):
        self.total_number = self.true_positive + self.true_negative + self.false_positive + self.false_negative

    def _gen_true_positive_rate(self):
        if (self.true_positive + self.false_negative) != 0:
            self.true_positive_rate = self.true_positive / \
                float(self.true_positive + self.false_negative)

    def _gen_true_negative_rate(self):
        if (self.true_negative + self.false_positive) != 0:
            self.true_negative_rate = self.true_negative / \
                float(self.true_negative + self.false_positive)

    def _gen_positive_predictive_value(self):
        if (self.true_positive + self.false_positive) != 0:
            self.positive_predictive_value = self.true_positive / \
                float(self.true_positive + self.false_positive)

    def _gen_negative_predictive_value(self):
        if (self.true_negative + self.false_negative) != 0:
            self.negative_predictive_value = self.true_negative / \
                float(self.true_negative + self.false_negative)

    def _gen_false_positive_rate(self):
        if(self.true_negative + self.false_positive) != 0:
            self.false_positive_rate = self.false_positive / \
                float(self.true_negative + self.false_positive)

    def _get_false_negative_rate(self):
        if(self.true_positive + self.false_negative) != 0:
            self.false_negative_rate = self.false_negative / \
                float(self.true_positive + self.false_negative)

    def _get_false_discovery_rate(self):
        if(self.true_positive + self.false_positive) != 0:
            self.false_discovery_rate = self.false_positive / \
                float(self.true_positive + self.false_positive)

    def _get_false_omission_rate(self):
        if(self.true_negative + self.false_negative) != 0:
            self.false_omission_rate = self.false_negative / \
                float(self.true_negative + self.false_negative)

    def _get_f1_score(self):
        if (self.positive_predictive_value + self.true_positive_rate) != 0:
            self.f1_score = 2.0 * (self.positive_predictive_value * self.true_positive_rate) / \
                (self.positive_predictive_value + self.true_positive_rate)


class BinaryClassificationEvaluator(object):
    """ this object can take two boolean data and generate a evalution result.
    """

    def __init__(self,
                 outcome_data,
                 condition_data,
                 match_time_tolerance_in_second=0.05,
                 default_value=False):
        start_at = time.time()
        if outcome_data.key != condition_data.key:
            raise ValueError('Comparing different keys, outcome key: {}, condition key: {}'.format(
                outcome_data.key, condition_data.key))

        if outcome_data.value_type != 'boolean':
            raise TypeError('outcome_data type({}) is not boolean'.format(outcome_data.value_type))

        if condition_data.value_type != 'boolean':
            raise TypeError('condition_data type({}) is not boolean'.format(
                condition_data.value_type))

        self._true_positive = 0
        self._true_negative = 0
        self._false_positive = 0
        self._false_negative = 0
        matched_data_frame = match_time_and_fill(outcome_data.data_frame,
                                                 condition_data.data_frame,
                                                 match_time_tolerance_in_second,
                                                 default_value)
        print('match_time_and_fill took {} s'.format(time.time() - start_at))
        self._gen_basic_evaluations(matched_data_frame)
        self.result = BinaryClassificationEvaluationResult(self._true_positive,
                                                           self._true_negative,
                                                           self._false_positive,
                                                           self._false_negative)

    def _prase_and_add_tp_fn_tn_fp(self, test_boolean, condition_boolean):
        if condition_boolean is True:
            if test_boolean is True:
                self._true_positive = self._true_positive + 1
            else:
                self._false_negative = self._false_negative + 1
        else:
            if test_boolean is False:
                self._true_negative = self._true_negative + 1
            else:
                self._false_positive = self._false_positive + 1

    def _gen_basic_evaluations(self, matched_data_frame):
        for _, row in matched_data_frame.iterrows():
            print('base: {} gt:{}'.format(row['test_value'], row['target_value']))
            self._prase_and_add_tp_fn_tn_fp(row['test_value'], row['target_value'])
