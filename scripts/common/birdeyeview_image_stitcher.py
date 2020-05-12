#!/usr/bin/env python

import cv2
import numpy as np
import math


class BirdEyeViewImageStitcher(object):
    def __init__(self, tr_imu_to_cam, imu_height, projection_matrix, src_shape, dst_shape, scale):
        self.dst_shape = dst_shape
        self.scale = scale
        self.mapx, self.mapy = self.generate_birdeye_view_map(
            tr_imu_to_cam, imu_height, projection_matrix, src_shape, dst_shape, scale)

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def generate_birdeye_view_map(self, tr_imu_to_cam, imu_height, projection_matrix, src_shape, dst_shape, scale):
        map_x = np.zeros((dst_shape[0], dst_shape[1]), dtype=np.float32)
        map_y = np.zeros((dst_shape[0], dst_shape[1]), dtype=np.float32)

        nx, ny = (dst_shape[1], dst_shape[0])
        x = range(nx)
        y = range(ny)
        xv, yv = np.meshgrid(x, y)
        bev_pts = np.ones([xv.size, 2])
        bev_pts[:, 0] = np.ravel(xv)
        bev_pts[:, 1] = np.ravel(yv)

        # map imu point to birdeye view image
        imu_pts = np.ones([xv.size, 4])
        imu_pts[:, 0] = (bev_pts[:, 0] - (dst_shape[1]/2.0)) / scale
        # flip the y, because imu's +x: forward, +y: left, and bev_points is +x: forward, +y:right
        imu_pts[:, 1] = -1.0 * (bev_pts[:, 1] - dst_shape[0]/2.0) / scale
        imu_pts[:, 2] = -imu_height

        camera_pts = imu_pts.dot(np.transpose(tr_imu_to_cam))
        # print "origin shape ", camera_pts.shape
        # only select z > 0
        mask = camera_pts[:, 2] > 0
        camera_pts = camera_pts[mask]
        bev_pts = bev_pts[mask]

        img_pts = camera_pts.dot(np.transpose(projection_matrix))
        # print img_pts.shape
        img_pts /= img_pts[:, 2:3]
        # only select point within the image range
        mask2 = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < src_shape[1]) & (
            img_pts[:, 1] >= 0) & (img_pts[:, 1] < src_shape[0])

        img_pts = img_pts[mask2]
        bev_pts = bev_pts[mask2]
        # print img_pts
        # print bev_pts
        for index in range(bev_pts.shape[0]):
            map_x[int(bev_pts[index, 1]), int(bev_pts[index, 0])] = img_pts[index, 0]
            map_y[int(bev_pts[index, 1]), int(bev_pts[index, 0])] = img_pts[index, 1]
        return map_x, map_y
