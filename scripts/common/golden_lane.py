#!/usr/bin/env python

import argparse
import cv2
import calib
import numpy as np
import euler
import sys
import project_points


def project_to_image(imu_to_image, x, y, z):
    project_point = imu_to_image.dot(np.array([x, y, z, 1]))
    if project_point[2] > 0:
        project_point /= project_point[2]
        project_point = project_point[:2].astype(np.int)
        return True, project_point
    else:
        project_point = project_point[:2].astype(np.int)
        return False, project_point


def draw_golden_lane(img, start, length, lane_width, curvature, imu_height, imu_to_image):

    marking_width = 0.1
    step = 0.1

    curve_left_1 = []
    curve_left_2 = []
    curve_right_1 = []
    curve_right_2 = []
    for x in np.arange(start, start+length, step):
        y = curvature * x**2
        project_point = None
        ret_left_1, project_point_left_1 = project_to_image(
            imu_to_image, x, y - lane_width / 2 - marking_width, -imu_height)
        ret_left_2, project_point_left_2 = project_to_image(
            imu_to_image, x, y - lane_width / 2 + marking_width, -imu_height)
        ret_right_1, project_point_right_1 = project_to_image(
            imu_to_image, x, y + lane_width / 2 - marking_width, -imu_height)
        ret_right_2, project_point_right_2 = project_to_image(
            imu_to_image, x, y + lane_width / 2 + marking_width, -imu_height)
        if ret_left_1 and ret_left_2 and ret_right_1 and ret_right_2:
            curve_left_1.append(project_point_left_1)
            curve_left_2.append(project_point_left_2)
            curve_right_1.append(project_point_right_1)
            curve_right_2.append(project_point_right_2)

    # reverse to make polygon connection correctly(head-tail-tiail-head)
    curve_left_2.reverse()
    curve_right_2.reverse()

    contours = np.array([curve_left_1 + curve_left_2,
                         curve_right_1 + curve_right_2], dtype=int)

    lane_img = np.zeros_like(img)
    # need at least 4 point to drill a contour
    if contours.shape[1] >= 4:
        cv2.fillPoly(lane_img, contours, (0, 165, 255))
        return cv2.addWeighted(img, 1., lane_img, 0.5, 0.)
    else:
        return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo_calib", type=str, help="path to stereo calibration file")
    parser.add_argument("--mono_calib", default='mono.yml', type=str,
                        help="path to mono calibration file")
    parser.add_argument("--width", default=3.85, type=float, help="lane width")
    parser.add_argument("--imu_height", default=1.335, type=float, help="imu height setting")
    parser.add_argument("--curvature", default=0.0, type=float, help="lane curvature")
    parser.add_argument("--length", default=50.0, type=float, help="lane curvature")
    parser.add_argument("--input", type=str, metavar='image', nargs='+',
                        help="input image file names or a list of image(.txt)")
    parser.add_argument("--out", default="golden-lane.png", type=str, help="output image file name")
    args = parser.parse_args()

    imu_to_image = None
    tr_cam_to_imu = None
    p_matrix = None
    if args.stereo_calib:
        camera = calib.Stereo(args.stereo_calib)
        tr_cam_to_imu = camera.Tr_cam_to_imu
        p_matrix = camera.P1
    else:
        camera = calib.Mono(args.mono_calib)
        tr_cam_to_imu = camera.Tr_cam_to_imu
        p_matrix = camera.P

    image_files = []
    if 'txt' in args.input:
        f = open(args.input, 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.rstrip()
            image_files.append(line)
    else:
        print image_files
        image_files = args.input

    n_col = 5
    n_row = int(len(image_files) / n_col) if (len(image_files) %
                                              n_col) == 0 else int(len(image_files) / n_col) + 1
    scale = 0.5
    rotation_res = 0.001745  # ~0.1 degree
    translation_res = 0.1  # 0.1 meter

    finish = False
    output_img = np.array([])
    img = cv2.imread(image_files[0])
    patch_image_size = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    output_img = np.zeros((patch_image_size[1]*n_row, patch_image_size[0]*n_col, 3), np.uint8)
    r, p, y, x, y, z = 0, 0, 0, 0, 0, 0
    while not finish:
        print 'add offset [roll, pitch yaw, x, y, z] [{}, {}, {}, {}, {}, {}]'.format(r, p, y, x, y, z)
        tr_offset = euler.tr_matrix([r, p, y, x, y, z])
        new_tr_cam_to_imu = tr_offset.dot(tr_cam_to_imu)
        new_tr_cam_to_imu[0, 3] = x + tr_cam_to_imu[0, 3]
        new_tr_cam_to_imu[1, 3] = y + tr_cam_to_imu[1, 3]
        new_tr_cam_to_imu[2, 3] = z + tr_cam_to_imu[2, 3]
        t = new_tr_cam_to_imu.ravel().tolist()
        print 'new tr_cam_to_imu'
        print'[{0}, {1}, {2}, {3},\n {4}, {5}, {6}, {7},\n {8}, {9}, {10}, {11},\n {12}, {13}, {14},    {15}]'.format(*t)
        imu_to_image = p_matrix.dot(np.linalg.inv(new_tr_cam_to_imu))
        for i, image_file in enumerate(image_files):
            img = cv2.imread(image_file)
            img_with_lane = draw_golden_lane(img, start=10.0, curvature=args.curvature, length=args.length,
                                             width=args.width, imu_height=args.imu_height, imu_to_image=imu_to_image)
            img_with_lane = cv2.resize(img_with_lane, patch_image_size,
                                       interpolation=cv2.INTER_AREA)
            row = int(i / n_col)
            col = int(i % n_col)
            roi = output_img[row*patch_image_size[1]                             :(row+1)*patch_image_size[1], col*patch_image_size[0]: (col+1) * patch_image_size[0]]
            cv2.addWeighted(roi, 0., img_with_lane, 1.0, 0, roi)
        cv2.imshow('golden_lane', output_img)
        key = cv2.waitKey(0)
        if key == ord('k'):
            r = r + rotation_res
        elif key == ord('K'):
            r = r - rotation_res
        elif key == ord('l'):
            p = p + rotation_res
        elif key == ord('L'):
            p = p - rotation_res
        elif key == ord(';'):
            y = y + rotation_res
        elif key == ord(':'):
            y = y - rotation_res
        elif key == ord(','):
            x = x + translation_res
        elif key == ord('<'):
            x = x - translation_res
        elif key == ord('.'):
            y = y + translation_res
        elif key == ord('>'):
            y = y - translation_res
        elif key == ord('/'):
            z = z + translation_res
        elif key == ord('?'):
            z = z - translation_res
        elif key == 27:
            finish = True
