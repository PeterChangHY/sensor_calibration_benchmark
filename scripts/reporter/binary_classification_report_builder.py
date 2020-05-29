#!/usr/bin/env python
""" binary classification report buider module"""
import os


import utils

from report_builder import ReportBuilder
from binary_classification_evaluation import BinaryClassificationEvaluator, BinaryClassificationEvaluationResult
from data_creator import generate_boolean_data
import data_plotter


class BinaryClassificationReportBuilder(ReportBuilder):
    """ binary classification report buider module"""

    def __init__(self, settings):
        ReportBuilder.__init__(self, settings)
        self.evaluation_list = None

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

        self.evaluation_list = self._process(matched_fileset_list)
        self._render()

    def _process(self, matched_fileset_list):
        evaluation_list = []
        all_baseline_evaluation = BinaryClassificationEvaluationResult()
        all_target_evaluation = BinaryClassificationEvaluationResult()
        for matched_fileset in matched_fileset_list:
            evaluation = {}
            ground_truth_data = generate_boolean_data(self.settings.key, matched_fileset['groud_truth_file'])
            baseline_data = generate_boolean_data(self.settings.key, matched_fileset['basline_file'])
            baseline_evaluator = BinaryClassificationEvaluator(baseline_data, ground_truth_data, match_time_tolerance_in_second=0.05, default_value=False)
            baseline_evaluation = baseline_evaluator.result
            all_baseline_evaluation.add(baseline_evaluation)
            target_data = None
            target_evaluation = BinaryClassificationEvaluationResult()
            if 'target_file' in matched_fileset:
                target_data = generate_boolean_data(self.settings.key, matched_fileset['target_file'])
                target_evaluator = BinaryClassificationEvaluator(target_data, ground_truth_data, match_time_tolerance_in_second=0.05, default_value=False)
                target_evaluation = target_evaluator.result
                all_target_evaluation.add(target_evaluation)

            evaluation['bag'] = ground_truth_data.bag
            evaluation['stats'] = self._gen_stats(baseline_evaluation, target_evaluation)

            outfile = os.path.join(self.settings.output_dir, self.settings.key + "_" + ground_truth_data.bag + '_binary_classification.png')
            data_plotter.generate_binary_plot(baseline_data, ground_truth_data, self.settings.key, outfile, target_data)
            evaluation['plot_path'] = os.path.relpath(outfile, self.settings.output_dir)

            evaluation_list.append(evaluation)

        overall_evaluation = {'bag': 'overall', 'stats': self._gen_stats(all_baseline_evaluation, all_target_evaluation), 'plot_path': None}
        evaluation_list.insert(0, overall_evaluation)

        return evaluation_list

    def _hightlight_lower_float(self, float_1, float_2):
        float_format = '{:.5f}'
        if abs(float_1 - float_2) > 1e-5:
            if float_1 > float_2:
                return float_format.format(float_1), self._highlight_value(float_format.format(float_2), 1, color='red')
            else:
                return self._highlight_value(float_format.format(float_1), 1, color='red'), float_format.format(float_2)
        return float_format.format(float_1), float_format.format(float_2)

    def _hightlight_higher_float(self, float_1, float_2):
        float_format = '{:.5f}'
        if abs(float_1 - float_2) > 1e-5:
            if float_1 > float_2:
                return self._highlight_value(float_format.format(float_1), 1, color='red'), float_format.format(float_2)
            else:
                return float_format.format(float_1), self._highlight_value(float_format.format(float_2), 1, color='red')
        return float_format.format(float_1), float_format.format(float_2)

    def _gen_stats(self, baseline_eval, target_eval):
        stats = []
        stats.append(['true_positive_rate',
                      baseline_eval.true_positive_rate,
                      target_eval.true_positive_rate])
        stats.append(['true_negative_rate',
                      baseline_eval.true_negative_rate,
                      target_eval.true_negative_rate])
        stats.append(['positive_predictive_value',
                      baseline_eval.positive_predictive_value,
                      target_eval.positive_predictive_value])
        stats.append(['negative_predictive_value',
                      baseline_eval.negative_predictive_value,
                      target_eval.negative_predictive_value])
        stats.append(['false_positive_rate',
                      baseline_eval.false_positive_rate,
                      target_eval.false_positive_rate])
        stats.append(['false_negative_rate',
                      baseline_eval.false_negative_rate,
                      target_eval.false_negative_rate])
        stats.append(['false_discovery_rate',
                      baseline_eval.false_discovery_rate,
                      target_eval.false_discovery_rate])
        stats.append(['false_omission_rate',
                      baseline_eval.false_omission_rate,
                      target_eval.false_omission_rate])
        return stats

    def _render(self):
        rows = []
        for evaluation in self.evaluation_list:
            row = self._render_template('binary_classification_report_chart_row', {
                'bag': evaluation['bag'],
                'stats': evaluation['stats'],
                'chart_path': evaluation['plot_path']})
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
