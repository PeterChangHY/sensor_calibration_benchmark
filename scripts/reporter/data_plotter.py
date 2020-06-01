#!/usr/bin/env python
"""this module cotains the plot of binary classification data"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def generate_binary_plot(baseline_data, ground_truth_data, plot_name, outfile, target_data=None):
    """
        generate plot from data_frame, dataframe should have the format [ [Unix_Time1, booelan1], [Unix_Time1, booelan2]....]
    """
    if baseline_data.key != ground_truth_data.key:
        raise ValueError('Comparing different keys, baseline key: {}, condition key: {}'.format(baseline_data.key, ground_truth_data.key))
    if target_data is not None and target_data.key != ground_truth_data.key:
        raise ValueError('Comparing different keys, baseline key: {}, condition key: {}'.format(target_data.key, ground_truth_data.key))

    def shift_time(df, time_shift):
        df['Unix_Time'] -= time_shift

    def plot(plot_obj, data, color, label, alpha, marker, markersize, linesytle='solid'):
        if data.shape[0] > 0:
            x_data = data['Unix_Time']
            y_data = []
            for value in data['Parsed_value']:
                if value:
                    y_data.append(1.0)
                else:
                    y_data.append(0.0)
        else:
            x_data = []
            y_data = []

        plot_obj.plot(x_data, y_data, drawstyle='steps-post', color=color, label=label, alpha=alpha, marker=marker, linestyle=linesytle, linewidth=1, markersize=markersize)

    # shift time
    if len(ground_truth_data.data_frame['Unix_Time'].tolist()) > 0:
        time_shift = ground_truth_data.data_frame['Unix_Time'].tolist()[0]
        shift_time(ground_truth_data.data_frame, time_shift)
        shift_time(baseline_data.data_frame, time_shift)
        if target_data is not None:
            shift_time(target_data.data_frame, time_shift)

    # Add a small padding around Y, so the graph doesn't get clipped.
    # we use 1.0 for true, 0.0 for false, so using 1.1 and -0.1 as y limit should be enough
    max_value = 1.1
    min_value = -0.1

    plt.clf()
    plt.suptitle(plot_name)
    ax = plt.gca()
    ax.set_ylim(min_value, max_value)
    ax.set_yticklabels(['Off', 'On'])
    ax.set_xlabel('relative Time(s), start from Unix Time: {:5f})'.format(time_shift))

    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])

    plot(plt, ground_truth_data.data_frame, color=(0.9, 0.4, 0.4), label='ground_truth', alpha=1.0, marker='x', markersize=4, linesytle='solid')
    plot(plt, baseline_data.data_frame, color=(0.2, 0.2, 0.9), label='baseline', alpha=0.8, marker='d', markersize=3, linesytle='dotted')
    if target_data is not None:
        plot(plt, target_data.data_frame, color=(0.1, 0.9, 0.1), label='target', alpha=0.8, marker='s', markersize=1, linesytle=(0, (5, 10)))

    plt.legend()
    plt.savefig(outfile, dpi=400)
    plt.clf()
