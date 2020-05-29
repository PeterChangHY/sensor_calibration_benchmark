#!/usr/bin/env python

import cv2
import numpy as np
import math


class CylindricalImageStitcher(object):
    def __init__(self, tr_cam_to_imu, projection_matrix, src_shape, dst_shape):
        self.dst_shape = dst_shape
        self.mapx, self.mapy = self.generate_cylindrical_view_map(
            tr_cam_to_imu, projection_matrix, src_shape, dst_shape)

    def remap(self, img):
        return cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def generate_cylindrical_view_map(self, tr_cam_to_imu, projection_matrix, src_shape, dst_shape):

        vertical_view_of_view_deg = 50
        map_x = np.zeros((dst_shape[0], dst_shape[1]), dtype=np.float32)
        map_y = np.zeros((dst_shape[0], dst_shape[1]), dtype=np.float32)

        h_, w_ = src_shape[:2]

        # pixel coordinates
        y_i, x_i = np.indices((h_, w_))

        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
        Pinv = np.linalg.inv(projection_matrix[:3, :3])

        homo_X = Pinv.dot(X.T).T  # normalized coords
        img_rays = np.concatenate([homo_X, np.zeros_like(x_i).reshape(h_ * w_, 1)], axis=1)

        imu_rays = img_rays.dot(np.transpose(tr_cam_to_imu))
        # calculate cylindrical coords
        # note: in imu's coordinate: +x is forward, +y is left, +z is up
        cylindrical_coords = np.zeros((h_ * w_, 2), dtype=np.float32)
        cylindrical_coords[:, 0] = (math.pi - np.arctan2(imu_rays[:, 1],
                                                         imu_rays[:, 0])) * (dst_shape[1] / (2 * math.pi))
        xylength = np.linalg.norm(imu_rays[:, 0:2], axis=1)
        cylindrical_coords[:, 1] = -1.0 * (np.arctan2(imu_rays[:, 2], xylength)) * (
            dst_shape[0] / (vertical_view_of_view_deg * math.pi / 180.0)) + dst_shape[0] / 2

        for index in range(h_ * w_):
            if cylindrical_coords[index, 0] >= 0 and cylindrical_coords[index, 0] < dst_shape[1] and cylindrical_coords[index, 1] >= 0 and cylindrical_coords[index, 1] < dst_shape[0]:
                map_x[int(cylindrical_coords[index, 1]), int(
                    cylindrical_coords[index, 0])] = X[index, 0]
                map_y[int(cylindrical_coords[index, 1]), int(
                    cylindrical_coords[index, 0])] = X[index, 1]
        return map_x, map_y
