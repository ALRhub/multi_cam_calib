#!/usr/bin/env python

import numpy as np 
import cv2
import multi_cam_calib_py.utils as utils 

class Detector:
    def __init__(self):
        pass

    def detect_contour(self, c):
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        return len(approx), peri, area

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(gray, 1.0, 250.0, cv2.THRESH_BINARY)

        cv2.imshow('img_thresh', utils.resize_with_aspect_ratio(img_thresh, 1500))
        cv2.waitKey(0)
        input("continue")
        
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []

        for c in contours:
            shape = self.detect_contour(c)

            if 14 > shape[0] > 10 and 800 > shape[1] > 50 and 15000 > shape[2] > 100:
                m = cv2.moments(c)

                c_x = int((m["m10"] / m["m00"]))
                c_y = int((m["m01"] / m["m00"]))
                centers.append(np.array([[c_x], [c_y]]))

        return centers
