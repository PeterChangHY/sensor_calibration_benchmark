#!/usr/bin/python
""" Tool to find imu-to-vehicle-yaw-offset """
import os
import math
import tf
import numpy as np
import argparse

import rosbag

csv_pattern = '%.4f,%s,%.1f\n'


class DataCollector(object):
    """
    Data collector
    """

    def __init__(self, sample_rate, yaw_th_in_rad=None, output_dir=None):
        self.imu_yaw = float('inf')
        self.x = float('inf')
        self.y = float('inf')
        self.x_last = float('inf')
        self.y_last = float('inf')
        self.imu_yaws = []
        self.velocity_yaws = []
        self.yaw_diffs_deg = []
        self.yaw_th_in_rad = yaw_th_in_rad
        self.sample_rate = sample_rate
        self.last_time = float('inf')
        self.write_file = None
        if output_dir:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            filebasename = 'imu-yaw-to-velocity-yaw-result.csv'
            self.write_file = open(os.path.join(output_dir, filebasename), 'w')

    def callback_odom(self, data):
        ox, oy, oz, ow = [
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        ]

        _, _, self.imu_yaw = tf.transformations.euler_from_matrix(
            tf.transformations.quaternion_matrix([ox, oy, oz, ow]))
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

        if self.x_last != float('inf') and self.y_last != float('inf'):
            if (data.header.stamp.to_sec() - self.last_time) > self.sample_rate:
                return
            x_diff = self.x - self.x_last
            y_diff = self.y - self.y_last
            if math.sqrt(x_diff * x_diff + y_diff * y_diff) > 1.0:
                velocity_yaw = math.atan2(y_diff, x_diff)
                yaw_offset_rad = (self.imu_yaw - velocity_yaw)
                yaw_offset_deg = yaw_offset_rad * 180 / math.pi
                self.imu_yaws += [self.imu_yaw]
                self.velocity_yaws += [velocity_yaw]
                self.yaw_diffs_deg += [yaw_offset_deg]
                print('============================================')
                print('Time stamp: %f' % data.header.stamp.to_sec())
                print("Imu_yaw:  cur %f, avg %f, std %f" %
                      (self.imu_yaw, np.mean(
                          self.imu_yaws), np.std(self.imu_yaws)))
                print("Velocity_yaw: cur %f, avg %f, std %f" %
                      (velocity_yaw, np.mean(self.velocity_yaws), np.std(
                          self.velocity_yaws)))
                print(
                    "The diff of imu_yaw_to_velocity_yaw(degree): cur %f, avg %f, std %f"
                    % (yaw_offset_deg, np.mean(
                        self.yaw_diffs_deg), np.std(self.yaw_diffs_deg)))
                print(
                    "The diff of imu_yaw_to_velocity_yaw(radian): cur %f, avg %f, std %f"
                    % (yaw_offset_rad, np.mean(
                        self.yaw_diffs_deg) * math.pi / 180, np.std(self.yaw_diffs_deg) * math.pi / 180))
                self.x_last = self.x
                self.y_last = self.y
                self.last_time = data.header.stamp.to_sec()

                if self.yaw_th_in_rad and self.write_file:
                    if abs(yaw_offset_rad) > self.yaw_th_in_rad:
                        result = csv_pattern % (data.header.stamp.to_sec(),
                                                'SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF', 1.0)
                        print("Write result: " + result)
                        self.write_file.write(result)

                    else:
                        result = csv_pattern % (data.header.stamp.to_sec(),
                                                'SENSOR_CALIB_EXTRINSIC_PARAM_IMU_OFF', 0.0)
                        print("Write result: " + result)
                        self.write_file.write(result)

        else:
            self.x_last = self.x
            self.y_last = self.y
            self.last_time = data.header.stamp.to_sec()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=str, help="path to bag")
    parser.add_argument("--sample_rate", default=0.1, type=float, help="bag sample rate in second")
    parser.add_argument("--odom_topic", default='/navsat/odom', type=str, help="topic of odometry")
    parser.add_argument("--yaw_th_in_rad", default=None, metavar='THRESHOLD', type=float,
                        help="yaw offset threshold in radian")
    parser.add_argument("--output_dir", default=None, metavar='PATH',
                        type=str, help="output directory of result")

    args = parser.parse_args()

    dc = DataCollector(args.sample_rate, args.yaw_th_in_rad, args.output_dir)
    ros_bag = rosbag.Bag(args.bag)
    topics = [args.odom_topic]

    for msg in ros_bag.read_messages(topics=topics):
        if msg.topic == args.odom_topic:
            dc.callback_odom(msg.message)

    print("Write results at: " + dc.write_file.name if dc.write_file else 'None')


if __name__ == '__main__':
    main()
