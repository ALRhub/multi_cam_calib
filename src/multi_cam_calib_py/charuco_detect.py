#!/usr/bin/env python

import rospy
import cv2 
import cv2.aruco as aruco
from sensor_msgs.msg import Image
import numpy as np


class CharucoCalibration:
    def __init__(self):
        # ********************************************************************************
        # These parameters need to be modified with respecto to available ChAruco board
        # ********************************************************************************
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard_create(3, 3, 0.055, (0.055/8*6), self.dictionary)
        # ********************************************************************************
        self.arucoParams = aruco.DetectorParameters_create()
        self.cam_mtx = None
        self.distance_params = None
        self.rvec = None
        self.tvec = None
        self.target_frame = None
        self.calibrated = False
        
    
    def charuco_calibration(self,images, cam_mtx, distance_params):
        """ Internal ChAruco Calibration
        param image: rgb image
        param cam_mtx: current camera matrix
        param distance_params: current distance parameters

        returns:
            calibrated: True if calibration successful
            ret: final reprojection error
            cam_mtx: new camera matrix
            distance_params: new distance parameters
            rvec: list of rotation vectors between camera and ChAruco board
            tvec: list of translation vectors between camera and ChAruco board
            corner_list: list of identified corner positions
            id_list: list of identified marker ids

        """
        self.calibrated = False
        rospy.loginfo("started calibration")
        corners_list,id_list = [],[]
        charuco_corners = None
        charuco_ids = None
        for image in images:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = aruco.detectMarkers(image_gray, self.dictionary, parameters=self.arucoParams)
            resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=image_gray, board=self.board)
            
            if resp > 2:
                corners_list.append(charuco_corners)
                id_list.append(charuco_ids)
            else:
                rospy.loginfo("no ChAruco board detected!")
                return False, [],[],[]

        rospy.loginfo("Calibrating camera")
        ret, self.cam_mtx, self.distance_params, _, _ = aruco.calibrateCameraCharuco(charucoCorners=corners_list, charucoIds=id_list, board=self.board, imageSize=image_gray.shape, cameraMatrix=cam_mtx, distCoeffs=distance_params)
        _, self.rvec, self.tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.mtx, self.dist, None, None)
        self.calibrated = True
        rospy.loginfo("Camera calibrated successfully")
        
        return self.calibrated, ret, self.mtx, self.dist, self.rvec.flatten().tolist(), self.tvec.flatten().tolist(), corners_list, id_list




    def charuco_calibration_ext(self, image, cam_mtx, distance_params):
        """ External ChAruco Calibration
        param image: rgb image
        param cam_mtx: current camera matrix
        param distance_params: current distance parameters

        returns:
            target_frame: image with drawn axis of detected ChAruco board
            rvec: list of rotation vectors between camera and ChAruco board
            tvec: list of translation vectors between camera and ChAruco board
        """
        self.cam_mtx = cam_mtx
        self.distance_params = distance_params
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image_gray, self.dictionary, parameters=self.arucoParams)
        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(markerCorners=corners, markerIds=ids, image=image_gray, board=self.board)
        
        if resp > 2:
            _, self.rvec, self.tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, self.cam_mtx, self.distance_params, self.rvec, self.tvec)

            self.target_frame = aruco.drawAxis(image, cam_mtx, distance_params, self.rvec, self.tvec, 0.1)

            return self.target_frame, self.rvec.flatten().tolist(), self.tvec.flatten().tolist()
        else:
            rospy.loginfo("no ChAruco board detected!")
            return image, None, None


if __name__ == "__main__":
    CharucoCalibration()
