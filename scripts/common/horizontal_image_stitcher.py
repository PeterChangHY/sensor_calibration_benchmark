#!/usr/bin/env python

import cv2
import numpy as np

# return the horizontal stacked combined image = [img0 img1 img2 ...] and reshape the whole image to self.dst_shape


class HorizontalImageStitcher(object):
    def __init__(self, height=1020):
        self.height = height

    def horizontal_stitch(self, images):
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

        return display_img
