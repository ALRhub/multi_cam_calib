#!/usr/bin/env python

import rospy
import cv2 
import cv2.aruco as aruco
import numpy as np


class CharucoCalibration:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard_create(7, 5, 0.075, (0.075/8*6), self.dictionary)
        self.arucoParams = aruco.DetectorParameters_create()
        self.mtx = None
        self.dist = None
        self.rvec = None
        self.tvec = None
        self.calibrated = False


    def charuco_calibration(self,images, cam_mtx, dist):
        self.calibrated = False
        rospy.loginfo("started calibration")
        corners_list,id_list = [],[]
        charuco_corners = None
        charuco_ids = None
        for image in images:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = aruco.detectMarkers(img_gray, self.dictionary, parameters=self.arucoParams)
            resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=img_gray, board=self.board)
            
            if resp > 20:
                corners_list.append(charuco_corners)
                id_list.append(charuco_ids)
            else:
                rospy.loginfo("no charuco board detected!")
                return False, [],[],[]

        rospy.loginfo("calibrating camera")
        ret, self.mtx, self.dist, _, _ = aruco.calibrateCameraCharuco(charucoCorners=corners_list, charucoIds=id_list, board=self.board, imageSize=img_gray.shape, cameraMatrix=cam_mtx, distCoeffs=dist)
        _, self.rvec, self.tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.mtx, self.dist, None, None)
        self.calibrated = True
        rospy.loginfo("calibrated camera")
        
        return self.calibrated, ret, self.mtx, self.dist, self.rvec.flatten().tolist(), self.tvec.flatten().tolist(), corners_list, id_list


    def charuco_calibration_ext(self, image, cam_mtx, dist):
        self.mtx = cam_mtx
        self.dist = dist
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(img_gray, self.dictionary, parameters=self.arucoParams)
        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=img_gray, board=self.board)
        
        if resp > 20:
            _, self.rvec, self.tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.mtx, self.dist, self.rvec, self.tvec)
            return self.rvec.flatten().tolist(), self.tvec.flatten().tolist()
        else:
            rospy.loginfo("no charuco board detected!")
            return None, None


if __name__ == "__main__":
    CharucoCalibration()
