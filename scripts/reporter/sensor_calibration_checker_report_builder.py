"""This module include the SensorCalibrationCheckerReportBuilder Class """
from binary_classification_report_builder import BinaryClassificationReportBuilder
import os


class SensorCalibrationCheckerReportBuilder(BinaryClassificationReportBuilder):
    """In sesnor calibration checker, we track two metrics
       1. positive_predictive_value aka precision: TP / (TP + FP)
       2. negative_predictive_value: TN / (TN + FN)
       The reasons why we choose these two are the following.
       1. We favor a high precision system
       2. Using negative_predictive_value instead of recall is
          because in the general data, the number of the positive case usally is very rare.
          Therefore, when calculating the recall(TP/(TP+FN)), we will meet the case that divied by 0 very often.
          That's why we don't use recall.
    """

    def __init__(self, settings):
        BinaryClassificationReportBuilder.__init__(self, settings)

    def _gen_stats(self, baseline_eval, target_eval):
        stats = []
        baseline_str_ppv, target_str_ppv = self._hightlight_lower_float(baseline_eval.positive_predictive_value, target_eval.positive_predictive_value)
        stats.append(['positive_predictive_value', baseline_str_ppv, target_str_ppv])

        baseline_str_npv, target_str_npv = self._hightlight_higher_float(baseline_eval.negative_predictive_value, target_eval.negative_predictive_value)
        stats.append(['negative_predictive_value', baseline_str_npv, target_str_npv])
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

        basefilename = self.settings.key + '_sensor_calibraiton_checker_report.html'
        filename = os.path.join(self.settings.output_dir, basefilename)
        with open(filename, 'wb') as report_file:
            report_file.write(res_content)
            report_file.write('\n')
        print('Generate report at: {}'.format(filename))
