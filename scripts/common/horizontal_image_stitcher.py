#!/usr/bin/env python

import cv2
import numpy as np

# return the horizontal stacked combined image = [img0 img1 img2 ...] and reshape the whole image to self.dst_shape


class HorizontalImageStitcher(object):
    def __init__(self, height=1020):
        self.height = height

    def horizontal_stitch(self, images, draw_epipolar_line=False):
        display_img = np.array([])
        for img in images:
            new_img = np.copy(img)
            h, w = img.shape[:2]
            aspect_ratio = float(w) / h
            if h != self.height:
                new_img = cv2.resize(new_img, (int(h * aspect_ratio), self.height),
                                     interpolation=cv2.INTER_AREA)

            if display_img.size == 0:
                display_img = np.copy(new_img)
            else:
                display_img = np.hstack((display_img, new_img))
        
        if draw_epipolar_line:
            height, width = display_img.shape[:2]
            color_red = (0,0,255)
            color_blue = (255,0,0)
            color_switch = False
            for h in range(0, height, 32):
                color_switch = not color_switch
                cv2.line(display_img, (0,h), (width,h), color=color_red if color_switch else color_blue)

        return display_img
