#!/usr/bin/env python


import rospy
import tf
from sensor_msgs.msg import Image, CameraInfo
from multi_cam_calib.msg import Calib
from cv_bridge import CvBridge
import cv2 
import numpy as np
import multi_cam_calib_py.charuco_detect as charuco_detect 
import multi_cam_calib_py.transformation as transformation 
import multi_cam_calib_py.utils as utils 


# Class for detecting and publishing transformation to ChAruco board
class CharucoDetector:
    def __init__(self):
        self.images=[]
        self.cam_mtx = None
        self.dst_params = None
        self.rvec = []
        self.tvec = []
        self.listener = None
        self.calibrated = rospy.get_param("~calibrated")
        self.undistort  = not rospy.get_param("~calibrated")
        self.charuco = charuco_detect.CharucoCalibration()
        self.pub = rospy.Publisher(rospy.get_param("~pub_name"), Calib, queue_size=1)
        self.img_pub = rospy.Publisher(rospy.get_param("~sub_name")+"/charuco_board", Image, queue_size=10)
        self.ir_sub_name = rospy.get_param("~ir_sub_name")
        self.d_sub_name = rospy.get_param("~depth_sub_name")
        self.sub_name = rospy.get_param("~sub_name")
        self.cam_name = rospy.get_param("~cam_name")
        self.cam_info = rospy.get_param("~cam_info")
        self.ref_pnt_name = rospy.get_param("~target_frame_name")
        self.camera_base_name = rospy.get_param("~base_frame_name")
        self.charuco_link_frame = rospy.get_param("~charuco_link_frame_name")
        self.br = CvBridge()
        self.tb = tf.TransformBroadcaster()

    def publish_calibration(self, params):
        """ Publishes calculated calibration to specified ROS topic
            Getting calibration parameters to publish
        """
        calib_data = Calib()
        calib_data.mtx = params[0].flatten().tolist()
        calib_data.dist = params[1].flatten().tolist()
        calib_data.rvecs = params[2]
        calib_data.tvecs = params[3]

        self.pub.publish(calib_data)

        rquat = transformation.transform_to_quat(params[2])
        rvec_tuple = (rquat[0],rquat[1], rquat[2], rquat[3])
        tvec_tuple = (params[3][0],params[3][1],params[3][2])
        self.tb.sendTransform(tvec_tuple,rvec_tuple, rospy.Time.now(), self.cam_name +"/charuco", self.cam_name + self.charuco_link_frame)

    def get_cam_params(self, data):
        """ Setting camera parameters from a ROS topic
            Getting data from camera_info topic
        """
        self.cam_mtx = np.array(data.K).reshape((3,3))
        self.dst_params = np.array(data.D)

    def callback(self, data):
        """ Main callback method when new message is recieved
            Getting the data from subscribed topic
        """
        current_frame = self.br.imgmsg_to_cv2(data)
        bgr_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)

        # Execute internal calibration if not calibrated, else external calibration
        if not self.calibrated:
            if len(self.images) % 10 >= 0:
                input("Position camera:")

            self.images.append(bgr_frame)

            if len(self.images) >= 100:
                rospy.loginfo("calibrating camera")
                self.calibrated, _, self.cam_mtx, self.dst_params, self.rvec, self.tvec, _, _ = self.charuco.charuco_calibration(self.images, self.cam_mtx, self.dst_params)

        else: 
            target_frame, self.rvec, self.tvec = self.charuco.charuco_calibration_ext(bgr_frame, self.cam_mtx, self.dst_params)

            if self.rvec is not None and self.tvec is not None:
                self.publish_calibration([self.cam_mtx, self.dst_params, self.rvec, self.tvec])

                tvec_charuco_base, rvec_charuco_base = self.listener.lookupTransform(self.cam_name + "/" + self.ref_pnt_name, self.cam_name + self.camera_base_name, rospy.Time(0))

                rvec_trans = (rvec_charuco_base[0],rvec_charuco_base[1],rvec_charuco_base[2],rvec_charuco_base[3])
                tvec_trans = (tvec_charuco_base[0],tvec_charuco_base[1],tvec_charuco_base[2])

                self.tb.sendTransform(tvec_trans,rvec_trans, rospy.Time.now(), self.cam_name + self.camera_base_name, self.ref_pnt_name)
                self.img_pub.publish(self.br.cv2_to_imgmsg(target_frame))

            if self.undistort: 
                h, w = bgr_frame.shape[:2]
                cam_mtx_new, _ = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.dst_params, (w,h), 1, (w,h))
                undst = cv2.undistort(bgr_frame, self.cam_mtx, self.dst_params, None, cam_mtx_new)
                bgr_frame = undst

    def receive_message(self):
        """
            Method for receiving messages from subscribed topic
        """
        self.listener = tf.TransformListener()
        rospy.Subscriber(self.cam_info, CameraInfo, self.get_cam_params)
        rospy.Subscriber(self.sub_name, Image, self.callback)

        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('~', anonymous=True)
    detector = CharucoDetector()
    detector.receive_message()
