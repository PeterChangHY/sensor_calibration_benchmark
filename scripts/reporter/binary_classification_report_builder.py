#!/usr/bin/env python
""" binary classification report buider module"""
import os

import time
import utils

from report_builder import ReportBuilder
from binary_classification_evaluation import BinaryClassificationEvaluator, BinaryClassificationEvaluationResult
from data_creator import generate_boolean_data
import data_plotter


class BinaryClassificationReportBuilder(ReportBuilder):
    """ binary classification report buider module"""

    def __init__(self, settings):
        ReportBuilder.__init__(self, settings)
        self.render_data_list = None

    def build(self):
        """
            1. parese the tsv files
            2. evaluate the data
            3. render the report.html
        """
        def gen_basename_dictionary(search_dir):
            return {os.path.basename(file): file for file in utils.walk_files(search_dir, suffix='.tsv')}

        ground_truth_files = gen_basename_dictionary(self.settings.ground_truth_dir)
        baseline_files = gen_basename_dictionary(self.settings.baseline_dir)
        target_files = None
        if self.settings.target_dir is not None:
            target_files = gen_basename_dictionary(self.settings.target_dir)
        # match frames by basename
        matched_fileset_list = []
        for basename in ground_truth_files:
            matched_fileset = {}
            if basename in baseline_files:
                matched_fileset['groud_truth_file'] = ground_truth_files[basename]
                matched_fileset['basline_file'] = baseline_files[basename]
            if target_files is not None and basename in target_files:
                matched_fileset['target_file'] = target_files[basename]
            if matched_fileset:
                matched_fileset_list.append(matched_fileset)
        self.render_data_list = self._process(matched_fileset_list)
        self._render()

    def _process(self, matched_fileset_list):
        render_data_list = []
        all_baseline_evaluation = BinaryClassificationEvaluationResult()
        all_target_evaluation = BinaryClassificationEvaluationResult()
        for matched_fileset in matched_fileset_list:
            start_at = time.time()
            render_data = {}
            print(matched_fileset['groud_truth_file'])
            ground_truth_data, gt_valid = generate_boolean_data(self.settings.key, matched_fileset['groud_truth_file'])
            
            baseline_data, baseline_valid = generate_boolean_data(self.settings.key, matched_fileset['basline_file'])
            if not gt_valid or not baseline_valid:
                continue
            print('generate_boolean_data took {} s'.format(time.time() - start_at))
            baseline_evaluator = BinaryClassificationEvaluator(baseline_data, ground_truth_data, match_time_tolerance_in_second=0.05, default_value=False)
            print('evaluation took {} s'.format(time.time() - start_at))
            baseline_evaluation = baseline_evaluator.result
            print("deded")
            print(baseline_evaluation.true_positive)
            print(baseline_evaluation.true_negative)
            print(baseline_evaluation.false_positive)
            print(baseline_evaluation.false_negative)
            all_baseline_evaluation.add(baseline_evaluation)
            target_data = None
            target_evaluation = BinaryClassificationEvaluationResult()
            if 'target_file' in matched_fileset:
                target_data = generate_boolean_data(self.settings.key, matched_fileset['target_file'])
                target_evaluator = BinaryClassificationEvaluator(target_data, ground_truth_data, match_time_tolerance_in_second=0.05, default_value=False)
                target_evaluation = target_evaluator.result
                all_target_evaluation.add(target_evaluation)

            render_data['bag'] = ground_truth_data.bag
            render_data['confusion_matrix'] = self._gen_confusion_matrix(baseline_evaluation, target_evaluation)
            render_data['ratios'] = self._gen_ratios(baseline_evaluation, target_evaluation)

            outfile = os.path.join(self.settings.output_dir, self.settings.key + "_" + ground_truth_data.bag + '_binary_classification.png')
            if self.settings.no_figure:
                render_data['plot_path'] = None
            else:
                data_plotter.generate_binary_plot(baseline_data, ground_truth_data, self.settings.key, outfile, target_data)
                render_data['plot_path'] = os.path.relpath(outfile, self.settings.output_dir)

            render_data_list.append(render_data)
            print('process a file took {} s'.format(time.time() - start_at))

        overall_render_data = {'bag': 'overall',
                              'ratios': self._gen_ratios(all_baseline_evaluation, all_target_evaluation),
                              'confusion_matrix': self._gen_confusion_matrix(all_baseline_evaluation, all_target_evaluation),
                              'plot_path': None}
        render_data_list.insert(0, overall_render_data)

        return render_data_list

    def _gen_confusion_matrix(self, baseline_eval, target_eval):
        baseline_tp_str, target_tp_str = self._hightlight_lower_int(baseline_eval.true_positive, target_eval.true_positive)
        baseline_tn_str, target_tn_str = self._hightlight_lower_int(baseline_eval.true_negative, target_eval.true_negative)
        baseline_fp_str, target_fp_str = self._hightlight_higher_int(baseline_eval.false_positive, target_eval.false_positive)
        baseline_fn_str, target_fn_str = self._hightlight_higher_int(baseline_eval.false_negative, target_eval.false_negative)

        return {'True Positive': {'baseline': baseline_tp_str, 'target': target_tp_str},
                'True Negative': {'baseline': baseline_tn_str, 'target': target_tn_str},
                'False Positive': {'baseline': baseline_fp_str, 'target': target_fp_str},
                'False Negative': {'baseline': baseline_fn_str, 'target': target_fn_str},
                'total_number': {'baseline': baseline_eval.total_number, 'target': target_eval.total_number}}

    def _gen_ratio(self, name, baseline_value, target_value):
        return {'name': name, 'baseline': baseline_value, 'target': target_value}

    def _gen_true_positive_rate(self, baseline_eval, target_eval):
        return self._gen_ratio('true_positive_rate<br>TP / (TP + FN)',
                                baseline_eval.true_positive_rate,
                                target_eval.true_positive_rate)

    def _gen_true_negative_rate(self, baseline_eval, target_eval):
        return self._gen_ratio('true_negative_rate<br>TN / (TN + FP)',
                                baseline_eval.true_negative_rate,
                                target_eval.true_negative_rate)

    def _gen_positive_predictive_value(self, baseline_eval, target_eval):
        return self._gen_ratio('positive_predictive_value<br>TP / (TP + FP)',
                                baseline_eval.positive_predictive_value,
                                target_eval.positive_predictive_value)

    def _gen_negative_predictive_value(self, baseline_eval, target_eval):
        return self._gen_ratio('negative_predictive_value<br>TN / (TN + FN)',
                                baseline_eval.negative_predictive_value,
                                target_eval.negative_predictive_value)

    def _gen_false_positive_rate(self, baseline_eval, target_eval):
        return self._gen_ratio('false_positive_rate<br>FP / (TN + FP)',
                                baseline_eval.false_positive_rate,
                                target_eval.false_positive_rate)

    def _gen_false_negative_rate(self, baseline_eval, target_eval):
        return self._gen_ratio('false_negative_rate<br>FN / (TP + FN)',
                                baseline_eval.false_negative_rate,
                                target_eval.false_negative_rate)

    def _gen_false_discovery_rate(self, baseline_eval, target_eval):
        return self._gen_ratio('false_discovery_rate<br>FP / (TP + FP)',
                                baseline_eval.false_discovery_rate,
                                target_eval.false_discovery_rate)

    def _gen_false_omission_rate(self, baseline_eval, target_eval):
        return self._gen_ratio('false_omission_rate<br>FN / (TN + FN)',
                                baseline_eval.false_omission_rate,
                                target_eval.false_omission_rate)

    def _gen_ratios(self, baseline_eval, target_eval):
        ratios = []

        ratios.append(self._gen_true_positive_rate(baseline_eval, target_eval))
        ratios.append(self._gen_true_negative_rate(baseline_eval, target_eval))
        ratios.append(self._gen_positive_predictive_value(baseline_eval, target_eval))
        ratios.append(self._gen_negative_predictive_value(baseline_eval, target_eval))
        ratios.append(self._gen_false_positive_rate(baseline_eval, target_eval))
        ratios.append(self._gen_false_negative_rate(baseline_eval, target_eval))
        ratios.append(self._gen_false_discovery_rate(baseline_eval, target_eval))
        ratios.append(self._gen_false_omission_rate(baseline_eval, target_eval))

        return ratios

    def _render(self):
        rows = []
        for render_data in self.render_data_list:
            row = self._render_template('binary_classification_report_chart_row', {
                'bag': render_data['bag'],
                'confusion_matrix': render_data['confusion_matrix'],
                'ratios': render_data['ratios'],
                'chart_path': render_data['plot_path']})
            rows.append(row)
        target_name = 'N/A'
        if self.settings.target_dir is not None:
            target_name = os.path.basename(self.settings.target_dir)

        res_content = self._render_template('binary_classification_report_chart.html', {
            'title': "{}:\n {} -> {}".format(self.settings.key, os.path.basename(self.settings.baseline_dir), target_name),
            'rows': rows,
        })

        basefilename = self.settings.key + '_binary_classification_report.html'
        filename = os.path.join(self.settings.output_dir, basefilename)
        with open(filename, 'wb') as report_file:
            report_file.write(res_content)
            report_file.write('\n')
        print('Genrate report at: {}'.format(filename))
