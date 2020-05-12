#!/usr/bin/env python

import cv2
import numpy as np
import calib
import roadmodel
import math


def generate_birdeye_view_map(tr_imu_to_cam, imu_height, projection_matrix, src_shape, dst_shape, scale):
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
    imu_pts[:, 0] = (bev_pts[:, 0] - dst_shape[1]/2.0) / scale
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


def generate_cylindrical_view_map(tr_cam_to_imu, projection_matrix, src_shape, dst_shape):

    vertical_view_of_view_deg = 50
    map_x = np.zeros((dst_shape[0], dst_shape[1]), dtype=np.float32)
    map_y = np.zeros((dst_shape[0], dst_shape[1]), dtype=np.float32)

    h_, w_ = src_shape[:2]

    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))

    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_*w_, 3)  # to homog
    Pinv = np.linalg.inv(projection_matrix[:3, :3])

    homo_X = Pinv.dot(X.T).T  # normalized coords
    img_rays = np.concatenate([homo_X, np.zeros_like(x_i).reshape(h_*w_, 1)], axis=1)

    imu_rays = img_rays.dot(np.transpose(tr_cam_to_imu))
    # calculate cylindrical coords
    # note: in imu's coordinate: +x is forward, +y is left, +z is up
    cylindrical_coords = np.zeros((h_*w_, 2), dtype=np.float32)
    cylindrical_coords[:, 0] = (math.pi - np.arctan2(imu_rays[:, 1],
                                                     imu_rays[:, 0])) * (dst_shape[1] / (2*math.pi))
    xylength = np.linalg.norm(imu_rays[:, 0:2], axis=1)
    cylindrical_coords[:, 1] = -1.0 * (np.arctan2(imu_rays[:, 2], xylength)) * (
        dst_shape[0] / (vertical_view_of_view_deg * math.pi / 180.0)) + dst_shape[0]/2

    for index in range(h_*w_):
        if cylindrical_coords[index, 0] >= 0 and cylindrical_coords[index, 0] < dst_shape[1] and cylindrical_coords[index, 1] >= 0 and cylindrical_coords[index, 1] < dst_shape[0]:
            map_x[int(cylindrical_coords[index, 1]), int(
                cylindrical_coords[index, 0])] = X[index, 0]
            map_y[int(cylindrical_coords[index, 1]), int(
                cylindrical_coords[index, 0])] = X[index, 1]
    return map_x, map_y


# return the horizontal stacked combined image = [img0 img1 img2 ...] and reshape the whole image to dsz
def horizontal_image_stitcher(, dsz=(3840, 1020)):
    if len(img_and_calibs) == 1:
        display_img = cv2.resize(img_and_calib[0]['img'], dsz, interpolation=cv2.INTER_AREA)
        return display_img

    display_img = np.array([])
    height = dsz[1]
    for img_and_calib in img_and_calibs:
        img = img_and_calib["img"]
        new_img = np.copy(img)
        h, w = img.shape[:2]
        aspect_ratio = float(w) / h
        if h != height:
            new_img = cv2.resize(new_img, (int(height*aspect_ratio), height),
                                 interpolation=cv2.INTER_AREA)

        if display_img.size == 0:
            display_img = np.copy(new_img)
        else:
            display_img = np.hstack((display_img, new_img))

    display_img = cv2.resize(display_img, dsz, interpolation=cv2.INTER_AREA)
    return display_img


def run_birdeye_view():

    front_left_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/front_left_camera/rect_0000.png"
    left_side_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/left_side_camera/rect_0000.png"
    right_side_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/right_side_camera/rect_0000.png"
    rear_left_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/rear_left_camera/rect_0000.png"
    rear_right_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/rear_right_camera/rect_0000.png"

    front_left_camrea_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200214_front_left_camera.yml"
    front_left_camrea_calib_path = "/home/peterchang/test/20200228_paccar-k001dm/paccar-k001dm_20200214_front_left_camera_op.yml"
    left_side_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_left_side_camera.yml"
    right_side_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_right_side_camera.yml"
    rear_left_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_rear_left_camera.yml"
    rear_right_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_rear_right_camera.yml"

    front_left_img = cv2.imread(front_left_img_path)
    left_side_img = cv2.imread(left_side_img_path)
    right_side_img = cv2.imread(right_side_img_path)
    rear_left_img = cv2.imread(rear_left_img_path)
    rear_right_img = cv2.imread(rear_right_img_path)

    front_left_calib = calib.Mono(front_left_camrea_calib_path)
    left_side_calib = calib.Mono(left_side_camera_calib_path)
    right_side_calib = calib.Mono(right_side_camera_calib_path)
    rear_left_calib = calib.Mono(rear_left_camera_calib_path)
    rear_right_calib = calib.Mono(rear_right_camera_calib_path)

    bev_img = np.zeros((height, width, 3), dtype=np.uint8)

    imu_height = 1.3
    dst_shape = [2000, 4000]
    front_left_img[0, 0, :] = [0, 0, 0]
    scale = 15.0
    map1, map2 = generate_birdeye_view_map(
        front_left_calib.Tr_imu_to_cam, imu_height, front_left_calib.P, front_left_img.shape, dst_shape, scale)
    dst_img_fl = cv2.remap(front_left_img, map1, map2, cv2.INTER_LINEAR)

    left_side_img[0, 0, :] = [0, 0, 0]
    map1, map2 = generate_birdeye_view_map(
        left_side_calib.Tr_imu_to_cam, imu_height, left_side_calib.P, left_side_img.shape, dst_shape, scale)
    dst_img_ls = cv2.remap(left_side_img, map1, map2, cv2.INTER_LINEAR)

    right_side_img[0, 0, :] = [0, 0, 0]
    map1, map2 = generate_birdeye_view_map(
        right_side_calib.Tr_imu_to_cam, imu_height, right_side_calib.P, right_side_img.shape, dst_shape, scale)
    dst_img_rs = cv2.remap(right_side_img, map1, map2, cv2.INTER_LINEAR)

    rear_left_img[0, 0, :] = [0, 0, 0]
    map1, map2 = generate_birdeye_view_map(
        rear_left_calib.Tr_imu_to_cam, imu_height, rear_left_calib.P, rear_left_img.shape, dst_shape, scale)
    dst_img_rl = cv2.remap(rear_left_img, map1, map2, cv2.INTER_LINEAR)

    rear_right_img[0, 0, :] = [0, 0, 0]
    map1, map2 = generate_birdeye_view_map(
        rear_right_calib.Tr_imu_to_cam, imu_height, rear_right_calib.P, rear_right_img.shape, dst_shape, scale)
    dst_img_rr = cv2.remap(rear_right_img, map1, map2, cv2.INTER_LINEAR)

    bev_img = cv2.addWeighted(bev_img, 1.0, dst_img_fl, 0.5, 0)
    bev_img = cv2.addWeighted(bev_img, 1.0, dst_img_ls, 0.5, 0)
    bev_img = cv2.addWeighted(bev_img, 1.0, dst_img_rs, 0.5, 0)
    bev_img = cv2.addWeighted(bev_img, 1.0, dst_img_rl, 0.5, 0)
    bev_img = cv2.addWeighted(bev_img, 1.0, dst_img_rr, 0.5, 0)

    # draw grid

    for i in range(-10, 15, 5):
        # horizontal line
        cv2.line(bev_img, (0, int(i*scale + height/2)),
                 (4000, int(i*scale + height/2)), (0, 0, 255), 1)
        # vertical line
        #cv2.line(bev_img, (i*scale + width/2,0), (i*scale + width/2,2000), color, thickness)
    for i in range(-100, 105, 5):
        # vertical line
        cv2.line(bev_img, (int(i*scale + width/2), 0),
                 (int(i*scale + width/2), 2000), (0, 0, 255), 1)

    cv2.putText(bev_img, "0,0", (int(width/2), int(height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "5m", (int(width/2), int(height/2 + 5*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "10m", (int(width/2), int(height/2 + 10*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "5m", (int(width/2 + 5 * scale), int(height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "50m", (int(width/2 + 50*scale), int(height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "100m", (int(width/2 + 100*scale), int(height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "-50m", (int(width/2 + -50*scale), int(height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.putText(bev_img, "-100m", (int(width/2 + -100*scale), int(height/2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    cv2.imshow("bev_img", bev_img)
    cv2.waitKey(0)


def run_cylindrical_view():
    front_left_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/front_left_camera/rect_0000.png"
    left_side_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/left_side_camera/rect_0000.png"
    right_side_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/right_side_camera/rect_0000.png"
    rear_left_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/rear_left_camera/rect_0000.png"
    rear_right_img_path = "/home/peterchang/test/20200228T160137_paccar-k001dm_0_0to2/rear_right_camera/rect_0000.png"

    front_left_camrea_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200214_front_left_camera.yml"
    front_left_camrea_calib_path = "/home/peterchang/test/20200228_paccar-k001dm/paccar-k001dm_20200214_front_left_camera_op.yml"
    left_side_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_left_side_camera.yml"
    right_side_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_right_side_camera.yml"
    rear_left_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_rear_left_camera.yml"
    rear_right_camera_calib_path = "/home/peterchang/myplusai/drive/perception/calib_db/paccar-k001dm_20200207_rear_right_camera.yml"

    front_left_img = cv2.imread(front_left_img_path)
    left_side_img = cv2.imread(left_side_img_path)
    right_side_img = cv2.imread(right_side_img_path)
    rear_left_img = cv2.imread(rear_left_img_path)
    rear_right_img = cv2.imread(rear_right_img_path)

    front_left_calib = calib.Mono(front_left_camrea_calib_path)
    left_side_calib = calib.Mono(left_side_camera_calib_path)
    right_side_calib = calib.Mono(right_side_camera_calib_path)
    rear_left_calib = calib.Mono(rear_left_camera_calib_path)
    rear_right_calib = calib.Mono(rear_right_camera_calib_path)

    dst_shape = [772, 3000]
    # hack to make
    front_left_img[0, 0, :] = [0, 0, 0]
    left_side_img[0, 0, :] = [0, 0, 0]
    rear_left_img[0, 0, :] = [0, 0, 0]
    right_side_img[0, 0, :] = [0, 0, 0]
    rear_right_img[0, 0, :] = [0, 0, 0]

    cylindrical_img = np.zeros((dst_shape[0], dst_shape[1], 3), dtype=np.uint8)

    map1, map2 = generate_cylindrical_view_map(
        front_left_calib.Tr_cam_to_imu, front_left_calib.P, front_left_img.shape, dst_shape)
    dst_img_fl = cv2.remap(front_left_img, map1, map2, cv2.INTER_LINEAR)

    map1, map2 = generate_cylindrical_view_map(
        left_side_calib.Tr_cam_to_imu, left_side_calib.P, left_side_img.shape, dst_shape)
    dst_img_ls = cv2.remap(left_side_img, map1, map2, cv2.INTER_LINEAR)

    map1, map2 = generate_cylindrical_view_map(
        rear_left_calib.Tr_cam_to_imu, rear_left_calib.P, rear_left_img.shape, dst_shape)
    dst_img_rl = cv2.remap(rear_left_img, map1, map2, cv2.INTER_LINEAR)

    map1, map2 = generate_cylindrical_view_map(
        right_side_calib.Tr_cam_to_imu, right_side_calib.P, right_side_img.shape, dst_shape)
    dst_img_rs = cv2.remap(right_side_img, map1, map2, cv2.INTER_LINEAR)

    map1, map2 = generate_cylindrical_view_map(
        rear_right_calib.Tr_cam_to_imu, rear_right_calib.P, rear_right_img.shape, dst_shape)
    dst_img_rr = cv2.remap(rear_right_img, map1, map2, cv2.INTER_LINEAR)

    cylindrical_img = cv2.addWeighted(cylindrical_img, 1.0, dst_img_fl, 0.8, 0)
    cylindrical_img = cv2.addWeighted(cylindrical_img, 1.0, dst_img_ls, 0.8, 0)
    cylindrical_img = cv2.addWeighted(cylindrical_img, 1.0, dst_img_rl, 0.8, 0)
    cylindrical_img = cv2.addWeighted(cylindrical_img, 1.0, dst_img_rs, 0.8, 0)
    cylindrical_img = cv2.addWeighted(cylindrical_img, 1.0, dst_img_rr, 0.8, 0)
    cv2.namedWindow("cylindrical_img", cv2.WINDOW_NORMAL)
    cv2.imshow("cylindrical_img", cylindrical_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    run_birdeye_view()
    run_cylindrical_view()
