#!/usr/bin/env python
import os
import stopclock
import munch
import utils as r_utils

from binary_classification_report_builder import BinaryClassificationReportBuilder
from sensor_calibration_checker_report_builder import SensorCalibrationCheckerReportBuilder


def main():
    import argparse
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True, type=str, metavar='DIR',
                        help="input directory contain baseline data(*.tsv)", dest='baseline_dir')
    parser.add_argument("--gt_dir", required=True, type=str, metavar='DIR',
                        help="input directory contain ground truth data(*.tsv)", dest='ground_truth_dir')
    parser.add_argument("--target_dir", required=False, type=str, metavar='DIR', default=None,
                        help="input directory contain target data(*.tsv)", dest='target_dir')
    parser.add_argument("--sync_tolerance", default=0.05, metavar='Second',
                        type=float, help="max time difference for synchronise")
    parser.add_argument("--key", required=True, type=str,
                        help="key to be analyzed")
    parser.add_argument("--mode", required=True, type=str,
                        choices=['binary_classification', 'sensor_calib_checker'], help="report mode")
    parser.add_argument("--no_figure", action='store_true', help="Don't genererate figures")
    parser.add_argument("--output_dir", default=None, metavar='DIR', dest='output_dir',
                        type=str, help="output directory of result")

    args = parser.parse_args()
    dirs = [args.ground_truth_dir, args.baseline_dir]
    if args.target_dir is not None:
        dirs = dirs + [args.target_dir]
    for p in dirs:
        if not os.path.isdir(p):
            raise RuntimeError("Could not find required"
                               " directory: {} (please ensure it exists)".format(p))

    template_dir, search_dirs = r_utils.find_dir('templates')
    if not template_dir:
        raise RuntimeError("Could not find required"
                           " template directory (please ensure it"
                           " exists), searched: %s" % search_dirs)

    settings = munch.Munch({
        'baseline_dir': os.path.abspath(args.baseline_dir),
        'ground_truth_dir': os.path.abspath(args.ground_truth_dir),
        'target_dir': os.path.abspath(args.target_dir) if args.target_dir is not None else None,
        'output_dir': os.path.abspath(args.output_dir),
        'key': args.key,
        'no_figure': args.no_figure,
        'template_dir': template_dir,
    })
    print("Settings:")
    print(settings)

    report_builders = {
        'binary_classification': BinaryClassificationReportBuilder,
        'sensor_calib_checker': SensorCalibrationCheckerReportBuilder,
    }

    report_builder_cls = report_builders[args.mode]
    report_builder = report_builder_cls(settings)

    with stopclock.Watch() as w:
        report_builder.build()
    print("All done, finished in %s seconds!" % w.elapsed())


if __name__ == '__main__':
    main()
