#!/usr/bin/env python

import rospy
import tf
import cv2 
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
#from multi_cam_calib.msg import Calib
from cv_bridge import CvBridge
import multi_cam_calib_py.marker_detect as detector
import multi_cam_calib_py.kalman_filter as kalman_filter
import multi_cam_calib_py.utils as utils 


class ImageReciever:
    def __init__(self):
        self.images=[]
        self.cam_mtx = None
        self.dst_params = None
        self.rvec = []
        self.tvec = []
        self.listener = None
        self.undistort  = not rospy.get_param("~calibrated")
        self.KF = kalman_filter.KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
        self.detector = detector.Detector()
        #self.pub = rospy.Publisher(rospy.get_param("~pub_name"), Calib, queue_size=1)
        self.ir_sub_name = rospy.get_param("~ir_sub_name")
        self.d_sub_name = rospy.get_param("~depth_sub_name")
        self.sub_name = rospy.get_param("~sub_name")
        self.cam_name = rospy.get_param("~cam_name")
        self.cam_info = rospy.get_param("~cam_info")
        self.ref_pnt_name = rospy.get_param("~trans_frame_name")
        self.br = CvBridge()
        self.tb = tf.TransformBroadcaster()
    

    def get_cam_params(self, data):
        self.cam_mtx = np.array(data.K).reshape((3,3))
        self.dst_params = np.array(data.D)


    def callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        print(current_frame.shape)
        bgr_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2BGR)

        if self.undistort: 
                h, w = bgr_frame.shape[:2]
                newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.dst_params, (w,h), 1, (w,h))
                undst = cv2.undistort(bgr_frame, self.cam_mtx, self.dst_params, None, newcameramtx)
                bgr_frame = undst

        centers = self.detector.detect(bgr_frame)

        if (len(centers) > 0):
            cv2.circle(bgr_frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            (x, y) = self.KF.predict()

            cv2.rectangle(bgr_frame, (x - 15, y - 15), (x + 15, y + 15), (255, 0, 0), 2)

            (x1, y1) = self.KF.update(centers[0])

            cv2.rectangle(bgr_frame, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 0, 255), 2)

            cv2.putText(bgr_frame, "Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(bgr_frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(bgr_frame, "Measured Position", (centers[0][0] + 15, centers[0][1] - 15), 0, 0.5, (0,191,255), 2)

        cv2.imshow(self.sub_name, utils.resize_with_aspect_ratio(bgr_frame, 1000))
        cv2.waitKey(2)


    def receive_message(self):
        self.listener = tf.TransformListener()
        rospy.Subscriber(self.cam_info, CameraInfo, self.get_cam_params)
        rospy.Subscriber(self.ir_sub_name, Image, self.callback)
        
        rospy.spin()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('~', anonymous=True)
    rec = ImageReciever()
    rec.receive_message()