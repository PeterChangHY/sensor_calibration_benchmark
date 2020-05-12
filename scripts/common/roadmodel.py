#!/usr/bin/env python

import numpy as np

def transform_points(pts, tr):
    # pts should be Nx3 np.array
    if len(pts.shape) < 2:
        pts = pts.reshape(1,3)
    pts_h = np.concatenate( (pts, np.ones((pts.shape[0],1),dtype=float)), axis=1) 
    transformed_pts = pts_h.dot(np.transpose(tr))
    return transformed_pts[:,:3]

class RoadModel:
    def __init__(self, tr_cam_to_imu, imu_height, projection_matrix):
        self.tr_cam_to_imu = tr_cam_to_imu
        self.imu_height = imu_height
        self.p = projection_matrix
        self.p_imu_to_img = self.p.dot(np.linalg.inv(tr_cam_to_imu))
        imu_to_img_ground = np.zeros((3,3), dtype=float)
        for i in range(3):
            imu_to_img_ground[i,0] = self.p_imu_to_img[i,0]
            imu_to_img_ground[i,1] = self.p_imu_to_img[i,1]
            imu_to_img_ground[i,2] = self.p_imu_to_img[i,3] +  self.p_imu_to_img[i,2] * -imu_height 
        self.imu_to_img_ground = imu_to_img_ground

    def imu_to_img(self, imu_pts):        
        # imu_point should be Nx3 np.array
        if len(imu_pts.shape) < 2:
            imu_pts = imu_pts.reshape(1,3)
        imu_pts_h = np.concatenate( (imu_pts, np.ones((imu_pts.shape[0],1),dtype=float)), axis=1) 
        img_pts = imu_pts_h.dot(np.transpose(self.p_imu_to_img))
        img_pts /= img_pts[:,2]
        return img_pts[:,:2]


    def img_to_imu(self, img_pts):
        # img_point should be Nx2 np.array
        if len(img_pts.shape) < 2:
            img_pts = img_pts.reshape(1,2)

        img_pts_h = np.concatenate((img_pts, np.ones((img_pts.shape[0],1),dtype=float)), axis=1) 
        img_pts_h = np.transpose(img_pts_h)
        imu_pts_h = np.linalg.solve(self.imu_to_img_ground,img_pts_h)
        imu_pts_h = np.transpose(imu_pts_h)
        imu_pts_h /= imu_pts_h[:,2:3]
        imu_pts = imu_pts_h
        imu_pts[:,2] = -self.imu_height
        return imu_pts

