"""base class for report builder"""
from __future__ import print_function
import pluspy.tpl_utils
import pluspy.utils

import os
import matplotlib
matplotlib.use('Agg')


class ReportBuilder(object):
    """base class for report builder"""

    def __init__(self, settings):
        self.settings = settings
        pluspy.utils.safe_make_dir(settings.output_dir)

    def _render_template(self, tpl_name, params, global_funcs=None):
        tpl_path = os.path.join(self.settings.template_dir, tpl_name)
        if not tpl_path.endswith(".j2"):
            tpl_path += ".j2"
        with open(tpl_path, 'rb') as fh:
            return pluspy.tpl_utils.render_template(fh.read(), params, globals=global_funcs)

    def _highlight_value(self, value, size=0, color='black'):
        font_str = "<font color=\"{}\" size=\"+{}\">".format(color, size) + \
                   str(value) + "</font>"
        return font_str

    def _hightlight_lower_float(self, float_1, float_2):
        float_format = '{:.3f}'
        if abs(float_1 - float_2) > 1e-5:
            if float_1 > float_2:
                return float_format.format(float_1), self._highlight_value(float_format.format(float_2), 1, color='red')
            else:
                return self._highlight_value(float_format.format(float_1), 1, color='red'), float_format.format(float_2)
        return float_format.format(float_1), float_format.format(float_2)

    def _hightlight_higher_float(self, float_1, float_2):
        float_format = '{:.3f}'
        if abs(float_1 - float_2) > 1e-5:
            if float_1 > float_2:
                return self._highlight_value(float_format.format(float_1), 1, color='red'), float_format.format(float_2)
            else:
                return float_format.format(float_1), self._highlight_value(float_format.format(float_2), 1, color='red')
        return float_format.format(float_1), float_format.format(float_2)

    def _hightlight_lower_int(self, int_1, int_2):
        int_format = '{}'
        if abs(int_1 - int_2) > 1e-5:
            if int_1 > int_2:
                return int_format.format(int_1), self._highlight_value(int_format.format(int_2), 1, color='red')
            else:
                return self._highlight_value(int_format.format(int_1), 1, color='red'), int_format.format(int_2)
        return int_format.format(int_1), int_format.format(int_2)

    def _hightlight_higher_int(self, int_1, int_2):
        int_format = '{}'
        if abs(int_1 - int_2) > 1e-5:
            if int_1 > int_2:
                return self._highlight_value(int_format.format(int_1), 1, color='red'), int_format.format(int_2)
            else:
                return int_format.format(int_1), self._highlight_value(int_format.format(int_2), 1, color='red')
        return int_format.format(int_1), int_format.format(int_2)

    def build(self):
        """ build the report"""
        raise NotImplementedError
