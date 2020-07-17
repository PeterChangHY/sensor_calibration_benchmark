"""This module include the SensorCalibrationCheckerReportBuilder Class """
from binary_classification_report_builder import BinaryClassificationReportBuilder
import os


class SensorCalibrationCheckerReportBuilder(BinaryClassificationReportBuilder):
    """In sesnor calibration checker, we track two metrics
       1. positive_predictive_value aka precision: TP / (TP + FP)
       2. negative_predictive_value: TN / (TN + FN)
       The reasons why we choose these two are the following.
       1. We favor a high precision system
       2. Using negative_predictive_value instead of recall is because in the general data,
       the number of the positive case usually is very rare.
       Therefore, when calculating the recall(TP/(TP+FN)), we will meet the case that divided by 0 very often.
       That's why we don't use recall
    """

    def __init__(self, settings):
        BinaryClassificationReportBuilder.__init__(self, settings)

    def _gen_ratios(self, baseline_eval, target_eval):
        ratios = []
        ppv_res = self._gen_positive_predictive_value(baseline_eval, target_eval)
        ppv_res['baseline'], ppv_res['target'] = self._hightlight_lower_float(ppv_res['baseline'], ppv_res['target'])
        ratios.append(ppv_res)

        npv_res = self._gen_negative_predictive_value(baseline_eval, target_eval)
        npv_res['baseline'], npv_res['target'] = self._hightlight_lower_float(npv_res['baseline'], npv_res['target'])
        ratios.append(npv_res)

        tnr_res = self._gen_true_negative_rate(baseline_eval, target_eval)
        tnr_res['baseline'], npv_res['target'] = self._hightlight_lower_float(tnr_res['baseline'], tnr_res['target'])
        ratios.append(tnr_res)

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

        basefilename = self.settings.key + '_sensor_calibraiton_checker_report.html'
        filename = os.path.join(self.settings.output_dir, basefilename)
        with open(filename, 'wb') as report_file:
            report_file.write(res_content)
            report_file.write('\n')
        print('Generate report at: {}'.format(filename))
