#!/usr/bin/env python

import sys
import numpy as np
import cv2
import pypcd
import calib
import euler
import math
import argparse


def point_cloud_from_path(pointcloud_path):
    cloud = pypcd.point_cloud_from_path(pointcloud_path)
    cloud_data = cloud.pc_data

    xyz_data = [ [data[0], data[1], data[2]] for data in cloud_data]
    xyz_data = np.array(xyz_data)
    print "pointcloud shape ", xyz_data.shape
    return xyz_data

def transform_point(tr, cloud):
    tr_points= []
    for row in range(cloud.shape[0]):
        point = cloud[row]
        point_augmented = np.array([point[0], point[1], point[2], 1])

        point_cam_3d = tr.dot(point_augmented)
        tr_points.append(point_cam_3d[:3])
    return np.array(tr_points)
def project_point(Tr_lidar_to_cam, p, image_size, cloud):
    proj_points= []
    for row in range(cloud.shape[0]):
        point = cloud[row]
        point_augmented = np.array([point[0], point[1], point[2], 1])

        point_cam_3d = Tr_lidar_to_cam.dot(point_augmented)
        point_cam_2d = p.dot(point_cam_3d)

        if point_cam_2d[2] <= 0:
            continue
        #print point_augmented
        #print point_cam_2d
        x = int(point_cam_2d[0] / point_cam_2d[2])
        y = int(point_cam_2d[1] / point_cam_2d[2])
        if x < 0 or x >= image_size[0]:
            continue
        if y < 0 or y >= image_size[1]:
            continue
        proj_points.append((x,y))
    
    return proj_points

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_calib", type=str, help="path to camrea calibration file")
    parser.add_argument("--lidar_calib", default='lidar.yml', type=str, help="path to lidar calibration file")
    parser.add_argument("--pointcloud_path", default="", type=str, help="path to pointcloud.pcd")
    parser.add_argument("--image_path", default="", type=str, help="path to image") 
    parser.add_argument("--output", default="project_points.png", type=str, help="output image file name")
    parser.add_argument("--verbose", default=False, type=bool, help="verbose logging")
    parser.add_argument("--yaw", default=0.0, type=float, help="")
    parser.add_argument("--pitch", default=0.0, type=float, help="")
    parser.add_argument("--roll", default=0.0, type=float, help="")
    args = parser.parse_args()


    lc = calib.Lidar(args.lidar_calib)
    Tr_lidar_to_imu = lc.Tr_lidar_to_imu
    print "Tr_lidar_to_imu:", Tr_lidar_to_imu
    Tr_imu_to_cam = np.eye(4)
    p = None

    image = cv2.imread(args.image_path)

    if calib.is_stereo_calib(args.camera_calib):
        sc = calib.Stereo(args.camera_calib)
        Tr_imu_to_cam = sc.Tr_imu_to_cam
        p = sc.P1
        image = sc.rectify_left(image)
    elif calib.is_mono_calib(args.camera_calib):
        mc = calib.Mono(args.camera_calib)
        Tr_imu_to_cam = mc.Tr_imu_to_cam
        p = mc.P
        image = mc.rectify(image)
    else:
        print args.camera_calib, ' is neither camera calib or stereo calib'
        exit()
    r = np.eye(4)
    r[:3,:3] = euler.r_matrix(args.roll, args.pitch, args.yaw)
    Tr_imu_to_cam = Tr_imu_to_cam.dot(r)
    print "new Tr_cam_to_imu: ", np.linalg.inv(Tr_imu_to_cam)
    Tr_lidar_to_cam = Tr_imu_to_cam.dot(Tr_lidar_to_imu)

    cloud = point_cloud_from_path(args.pointcloud_path)
    if args.verbose:
        print "pointcloud size: " , cloud.shape[0]


    for row in range(cloud.shape[0]):
        point = cloud[row]
        point_augmented = np.array([point[0], point[1], point[2], 1])

        point_cam_3d = Tr_lidar_to_cam.dot(point_augmented)
        point_cam_2d = p.dot(point_cam_3d)
        if args.verbose:
            print "point cam 3d: " , point_cam_3d, " \npoint cam 2d: ", point_cam_2d

        if point_cam_2d[2] <= 0:
            continue
        #print point_augmented
        #print point_cam_2d
        x = int(point_cam_2d[0] / point_cam_2d[2])
        y = int(point_cam_2d[1] / point_cam_2d[2])
        if x < 0 or x >= image.shape[1]:
            continue
        if y < 0 or y >= image.shape[0]:
            continue

        # image[y, x, 0] = 0
        # image[y, x, 0] = 255
        # image[y, x, 0] = 0
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    output = args.output
    cv2.imwrite(output, image)
    print "write projecting points at " , output
