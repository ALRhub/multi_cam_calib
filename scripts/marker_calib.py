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
        self.cam_mtx_d = None
        self.dst_params_d = None
        self.rvec = []
        self.tvec = []
        self.listener = None
        self.undistort  = not rospy.get_param("~calibrated")
        self.KF = kalman_filter.KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
        self.detector = detector.Detector()
        self.ir_sub_name = rospy.get_param("~ir_sub_name")
        self.d_sub_name = rospy.get_param("~depth_sub_name")
        self.sub_name = rospy.get_param("~sub_name")
        self.cam_name = rospy.get_param("~cam_name")
        self.cam_info = rospy.get_param("~cam_info")
        self.cam_info_d = rospy.get_param("~cam_info_depth")
        self.ref_pnt_name = rospy.get_param("~trans_frame_name")
        self.br = CvBridge()
        self.tb = tf.TransformBroadcaster()
        self.depth = None


    def get_cam_params_depth(self, data):
        self.cam_mtx_d = np.array(data.K).reshape((3,3))
        self.dst_params_d = np.array(data.D)    


    def get_cam_params(self, data):
        self.cam_mtx = np.array(data.K).reshape((3,3))
        self.dst_params = np.array(data.D)


    def get_depth_image(self, data):
        self.depth = self.br.imgmsg_to_cv2(data)
        if self.undistort: 
                h, w = self.depth.shape[:2]
                newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.dst_params, (w,h), 1, (w,h))
                undst = cv2.undistort(self.depth, self.cam_mtx, self.dst_params, None, newcameramtx)
                self.depth = undst

    def find_depth(self, depth_img, x, y):
        depth = depth_img[x, y]
        coordinates = x, y, depth

        return coordinates

    def callback(self, data):
        print("GotImage")
        current_frame = self.br.imgmsg_to_cv2(data)
        ir_frame = (current_frame/256).astype('uint8') 

        if self.undistort: 
                h, w = ir_frame.shape[:2]
                newcameramtx, _ = cv2.getOptimalNewCameraMatrix(self.cam_mtx, self.dst_params, (w,h), 1, (w,h))
                undst = cv2.undistort(ir_frame, self.cam_mtx, self.dst_params, None, newcameramtx)
                ir_frame = undst

        centers = self.detector.detect(ir_frame)

        if (len(centers) > 0):
            print(f"found {len(centers)} centers")
            cv2.circle(ir_frame, (int(centers[0][0]), int(centers[0][1])), 10, (255, 255, 255), 1)

            #x, y = self.KF.predict()
            #x, y = int(x), int(y)

            #cv2.rectangle(ir_frame, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 255), 1)

            #x1, y1 = self.KF.update(centers[0])
            #x1, y1 = int(x1), int(y1)

            #cv2.rectangle(ir_frame, (x1 - 10, y1 - 10), (x1 + 10, y1 + 10), (255, 255, 255), 1)

            #cv2.putText(ir_frame, "Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (255, 255, 255), 2)
            #cv2.putText(ir_frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(ir_frame, "Marker Position", (int(centers[0][0]) + 15, int(centers[0][1]) - 15), 0, 0.5, (255,255,255), 1)
            coordinates = self.find_depth(self.depth, int(centers[0][0]), int(centers[0][1]))

        cv2.imshow(self.sub_name, utils.resize_with_aspect_ratio(ir_frame, 1000))
        cv2.waitKey(2)


    def receive_message(self):
        self.listener = tf.TransformListener()
        rospy.Subscriber(self.cam_info, CameraInfo, self.get_cam_params)
        rospy.Subscriber(self.cam_info_d, CameraInfo, self.get_cam_params_depth)
        rospy.Subscriber(self.d_sub_name, Image, self.get_depth_image)
        rospy.Subscriber(self.ir_sub_name, Image, self.callback)
        
        rospy.spin()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('~', anonymous=True)
    rec = ImageReciever()
    rec.receive_message()