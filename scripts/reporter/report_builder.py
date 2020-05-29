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

    def build(self):
        """ build the report"""
        raise NotImplementedError
