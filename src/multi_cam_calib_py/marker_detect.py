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
        img_edges = cv2.Canny(frame,  50, 190, 3)

        ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)


        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_radius_thresh= 3
        max_radius_thresh= 30   

        centers = []

        for c in contours:
            
            shape = self.detect_contour(c)
            # if 14 > shape[0] > 10 and 800 > shape[1] > 50 and 15000 > shape[2] > 100:
            #     m = cv2.moments(c)

            #     c_x = int((m["m10"] / m["m00"]))
            #     c_y = int((m["m01"] / m["m00"]))
            #     centers.append(np.array([c_x, c_y]))

            (x, y), radius = cv2.minEnclosingCircle(c)
            radius = int(radius)
            
            print(x, y, radius)

            #Take only the valid circles
            # if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
             
        return centers
