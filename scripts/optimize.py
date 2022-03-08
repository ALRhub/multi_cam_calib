#!/usr/bin/env python
from distutils.command.config import config
from tokenize import String
import tf
import cma  # import CMA-ES optimization algorithm
import sys
import time
import rospy
import numpy as np
import moveit_commander
import matplotlib.pyplot as plt
import yaml
import transforms3d as t3d
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.affines import compose, decompose
from transforms3d.euler import quat2euler, mat2euler
import cv2
from sklearn.model_selection import train_test_split

# from copy import deepcopy
# from pprint import pprint as pp
# 
# 
# ####### LOG:
# - fixing planning frame panda_hand/panda_ee confict
# - adding rotations errors to SEE to force converging to correct solution (multiple correct solutions likely when monitoring translation errors only)
# - reviewed how q1 to q4 are calculated from rotation matrix
# - storing data in a dictionary to reduce the no of lists
# - saving data and prompting user if new data needs to be collected by running the robot
# - reduced goal tolerance callibration to 5 mm
# 
# ####### TO DO:
# - adding IF statement to only append data to list if marker detection = success  (marker likely to be moved)
# - fixing issue with loading and saving robot positions in yaml file

def ros2mat(trans, quat):
    return compose(trans, quat2mat([quat[3], quat[0], quat[1], quat[2]]), [1]*3)


class Calibration():
    AVAILABLE_ALGORITHMS = {
        'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    def __init__(self, cam_id, load_from_file = False):
        self.load_from_file = load_from_file

        self.tf_L0_EE = []
        self.tf_Cam_Calib = []
        
        self.val_tf_L0_EE = []
        self.val_tf_Cam_Calib = []


        if not load_from_file:
            self.listener = tf.TransformListener()
            self.br = tf.TransformBroadcaster()

        self.cam_id = cam_id

        self.tf_hand_to_board = self.trans_quat_to_mat([-0.0825, 0.0285, 0.2669], [0.5, 0.5, 0.5, 0.5])
        print(self.tf_hand_to_board)

    ########################
    ### Helper Functions ###
    ########################

    def get_transform_mat(self, from_tf, to_tf):
        """
            Getting the transforms from 'from_tf' to 'to_tf'
            returns 4x4 Matrix
        """
        assert self.load_from_file == False, "ROS INTERACTION NOT WORKING IF LOADING FROM FILE"

        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(5.0))
        trans, rot = self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))
        return ros2mat(trans, rot)


    def get_transform(self, from_tf, to_tf):
        """
            Getting the transforms from 'from_tf' to 'to_tf'
            returns [x,y,z][x,y,z,w]
        """
        assert self.load_from_file == False, "ROS INTERACTION NOT WORKING IF LOADING FROM FILE"

        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(10.0))
        return self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))

    def trans_quat_to_mat(self, trans, rot):
        rot_m = quat2mat(rot)
        trans_mat = np.concatenate((rot_m, np.array(trans).reshape((3, 1))), axis=1)
        trans_mat = np.concatenate((trans_mat, np.array([0, 0, 0, 1]).reshape((1,4))), axis=0)

        return trans_mat

    def myprint(self, to_print):
        sys.stdout.write(to_print)
        sys.stdout.flush()
        

    def _get_opencv_samples(self):
        """
            Returns the sample list as a rotation matrices and a translation vectors.
        """
        trans_EE_L0 = []
        rot_EE_L0 = []
        trans_cam_calib = []
        rot_cam_calib = []
        
        for tf_L0_EE, tf_cam_calib in zip(self.tf_L0_EE, self.tf_Cam_Calib):
            T, R, _, _ = decompose(np.linalg.inv(tf_L0_EE))
            trans_EE_L0.append(T)
            rot_EE_L0.append(R)
            
            T, R, _, _ = decompose(tf_cam_calib)
            trans_cam_calib.append(T)
            rot_cam_calib.append(R)
            
        return (rot_EE_L0, trans_EE_L0), (rot_cam_calib, trans_cam_calib)


    def save_yaml(self, file_name):
        with open(file_name, 'w') as stream:
            data_dict = {"tf_L0_EE": [n.tolist() for n in self.tf_L0_EE], "tf_Cam_Calib": [n.tolist() for n in self.tf_Cam_Calib]}
            yaml.dump(data_dict, stream)


    def load_yaml(self, file_name):
        with open(file_name, 'r') as stream:
            data_dict = yaml.safe_load(stream)
        
        self.tf_L0_EE = [np.array(tf) for tf in data_dict["tf_L0_EE"]]
        self.tf_Cam_Calib = [np.array(tf) for tf in data_dict["tf_Cam_Calib"]]
        

    ########################
    ### Gather Functions ###
    ########################

    def gather(self):
        """
            Gathering the current Transforms
            Specifically from Robot Base to EndEffector, and from Camera to CalibBoard
        """
        self.tf_L0_EE.append(self.get_transform_mat('panda_link0', 'panda_hand'))

        trans_cam_calib_list = []
        rot_cam_calib_list = []

        for _ in range(10):
            self.myprint(".")

            from_string = "cam_%i/camera_base" % self.cam_id
            to_string = "cam_%i/calib_board_small" % self.cam_id
            (trans, rot) = self.get_transform(from_string, to_string)
            
            trans_cam_calib_list.append(trans)
            rot_cam_calib_list.append(rot)

        trans_cam_calib_mean = np.mean(np.array(trans_cam_calib_list), axis=0)
        rot_cam_calib_mean = np.mean(np.array(rot_cam_calib_list), axis=0)


        self.tf_Cam_Calib.append(ros2mat(trans_cam_calib_mean, rot_cam_calib_mean))

    def separate_validation_set(self):
        self.tf_L0_EE, self.val_tf_L0_EE, self.tf_Cam_Calib, self.val_tf_Cam_Calib = train_test_split(self.tf_L0_EE, self.tf_Cam_Calib, test_size=0.33, random_state=42)

    ############################
    ### Validation Functions ###
    ############################

    def validate(self, tf_cam_L0):
        """
            Applying the estimated transformations and checking the variation on the combined transformation
        """
        trans_calib_calib_list = []
        rot_calib_calib_list = []
        for tf_L0_EE, tf_cam_calib in zip(self.val_tf_L0_EE, self.val_tf_Cam_Calib):
            tf_calib_cam = np.linalg.inv(tf_cam_calib)
            tf_calib_L0 = np.dot(tf_calib_cam, tf_cam_L0)
            tf_calib_EE = np.dot(tf_calib_L0, tf_L0_EE)
            tf_calib_calib = np.dot(tf_calib_EE, self.tf_hand_to_board)
            # tf_EE_Calib = np.dot(np.dot(tf_calib_cam, tf_cam_L0), tf_L0_EE)
        
            T, R, _, _ = decompose(tf_calib_calib)
            trans_calib_calib_list.append(T)
            rot_calib_calib_list.append(mat2euler(R))
            
        trans_error = np.linalg.norm(trans_calib_calib_list, axis=1) * 1000
        rot_error = np.linalg.norm(rot_calib_calib_list, axis=1)

        print "Translation Mean Err", np.mean(trans_error)
        print "Translation Err Variation", np.var(trans_error)
        print "Translation Max Err", np.max(trans_error)
        print "Translation Min Err", np.min(trans_error)
        print
        print "Rotation Mean Err", np.mean(rot_error)
        print "Rotation Err Variation", np.var(rot_error)
        print "Rotation Max Err", np.max(rot_error)
        print "Rotation Min Err", np.min(rot_error)
        print


    #####################
    ### Optimizations ###
    #####################

    def optimize_tsai(self):
        return self._optimize_opencv('Tsai-Lenz')
    
    def optimize_daniilidis(self):
        return self._optimize_opencv('Daniilidis')
    
    def optimize_horaud(self):
        return self._optimize_opencv('Horaud')

    def optimize_park(self):
        return self._optimize_opencv('Park')

    def optimize_andreff(self):
        return self._optimize_opencv('Andreff')
    
    def _optimize_opencv(self, algorithm):
        """
            Optimizing Transformation using OpenCV default methods
        """
        # Update data
        opencv_samples = self._get_opencv_samples()
        (hand_world_rot, hand_world_tr), (marker_camera_rot, marker_camera_tr) = opencv_samples

        method = self.AVAILABLE_ALGORITHMS[algorithm]
        hand_camera_rot, hand_camera_tr = cv2.calibrateHandEye(hand_world_rot, hand_world_tr, marker_camera_rot,
                                                               marker_camera_tr, method=method)
        result = compose(np.squeeze(hand_camera_tr), hand_camera_rot, [1, 1, 1])

        return np.linalg.inv(result) # Expecting cam_L0

    def optimize_cma_es(self):
        res = cma.fmin(self.objective_function_cma_es, [0.1]*7, 0.2)
        trans = res[0][0:3]
        quat = res[0][3:]
        quat = quat / np.linalg.norm(quat)

        return compose(trans, quat2mat(quat), [1]*3)

    def objective_function_cma_es(self, x):
        trans = [x[0], x[1], x[2]]
        rot = [x[3], x[4], x[5], x[6]]
        tf_est_cam_L0 = self.trans_quat_to_mat(trans, rot)
        pos_list = np.zeros((len(self.tf_L0_EE),3))
        orient_list = np.zeros((len(self.tf_L0_EE),4))

        for i in range(len(self.tf_L0_EE)):
            tf_L0_EE = self.tf_L0_EE[i]
            tf_cam_calib = self.tf_Cam_Calib[i]
            tf_calib_cam = np.linalg.inv(tf_cam_calib)
            tf_calib_EE = np.dot(np.dot(tf_calib_cam, tf_est_cam_L0), tf_L0_EE)
            pos_list[i,:] = tf_calib_EE[0:3,3]
            orient_list[i, :] = mat2quat(tf_calib_EE[0:3, 0:3])     #accepts full transformation matrix
        sse = np.sum(np.var(pos_list, axis=0)) #+ np.sum(np.var(orient_list, axis=0))

        return sse

    def optimize_cma_es_direct(self):
        res = cma.fmin(self.objective_function_cma_es_direct, [0.1]*7, 0.2)
        trans = res[0][0:3]
        quat = res[0][3:]
        quat = quat / np.linalg.norm(quat)

        return np.linalg.inv(compose(trans, quat2mat(quat), [1]*3))

    def objective_function_cma_es_direct(self, x):
        trans = [x[0], x[1], x[2]]
        rot = [x[3], x[4], x[5], x[6]]
        T = self.trans_quat_to_mat(trans, rot)

        SSE = 0

        for i in range(0, len(self.tf_L0_EE)):
            tf_L0_calib = np.dot(self.tf_L0_EE[i], self.tf_hand_to_board)
            Xi = tf_L0_calib[:3,3]
            tf_cam_calib = self.tf_Cam_Calib[i]
            Yi = tf_cam_calib[:,3]

            # temp = T.dot(Yi)
            temp = np.dot(T, Yi)
            SSE = SSE + np.sum(np.square(Xi - temp[0:3])) # Sum of Square Error (SSE)

        return SSE
        

    def optimize_cma_es_fulltf(self):
        res = cma.fmin(self.objective_function_cma_es_fulltf, [0.1]*7, 0.2)
        trans = res[0][0:3]
        quat = res[0][3:]
        quat = quat / np.linalg.norm(quat)

        return compose(trans, quat2mat(quat), [1]*3)

    def objective_function_cma_es_fulltf(self, x):
        trans = [x[0], x[1], x[2]]
        rot = [x[3], x[4], x[5], x[6]]
        T = self.trans_quat_to_mat(trans, rot)

        SSE = 0

        for i in range(0, len(self.tf_L0_EE)):
            tf_cam_calib = self.tf_Cam_Calib[i]
            tf_L0_EE = self.tf_L0_EE[i]
            
            tf_calib_cam = np.linalg.inv(tf_cam_calib)
            tf_calib_L0 = np.dot(tf_calib_cam, T)
            tf_calib_EE = np.dot(tf_calib_L0, tf_L0_EE)
            tf_calib_calib = np.dot(tf_calib_EE, self.tf_hand_to_board)
            # tf_EE_Calib = np.dot(np.dot(tf_calib_cam, tf_cam_L0), tf_L0_EE)
        
            trans, rot, _, _ = decompose(tf_calib_calib)
            SSE += np.square(np.linalg.norm(trans))

        return np.sqrt(SSE)

class Robot():
    def __init__(self):
        print("============ Initialising...")
        moveit_commander.roscpp_initialize(sys.argv)
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")
        self.arm.set_planner_id("FMTkConfigDefault")
        rospy.sleep(2)
        self.arm.set_end_effector_link("panda_hand")    # planning wrt to panda_hand or link8
        self.arm.set_max_velocity_scaling_factor(0.15)  # scaling down velocity
        self.arm.set_max_acceleration_scaling_factor(0.15)  # scaling down velocity
        self.arm.allow_replanning(True)
        self.arm.set_num_planning_attempts(10)
        self.arm.set_goal_position_tolerance(0.005)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_planning_time(10)

    def move(self):
        print("============ Moving...")
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

    def set_joint_positions(self, joints):
        self.arm.set_joint_value_target(joints)


def main():

    if len(sys.argv) < 2:
        print("DID NOT ENTER A CAMERA ID. Syntax: \"python optimize.py <camera_id>\"")
        return
    
    cam_id = int(sys.argv[1])

    if len(sys.argv) == 3:

        in_filename = sys.argv[2]
        print("Loading files from %s" % in_filename)

        calib = Calibration(cam_id, load_from_file=True)
        calib.load_yaml(in_filename)

    else:
        rospy.init_node('test', anonymous=True)
        robot = Robot()
        calib = Calibration(cam_id)


        filename = raw_input('Enter the filename with the joint_positions: ')

        if not filename: 
            filename = 'joint_states.yaml'

        with open(filename, 'r') as joint_states_file:
            joint_states = yaml.load(joint_states_file)
        
        for joint_positions in joint_states:
            print(joint_positions)
            robot.set_joint_positions(joint_positions)
            robot.move()
            time.sleep(2)
            calib.gather()
            time.sleep(1)

            # Saving current data at every step, because why not
            param_fle_name = "param_file_%i.yaml" % cam_id   
            calib.save_yaml(param_fle_name)

    calib.separate_validation_set()
    
    result_tsai       = calib.optimize_tsai()
    result_daniilidis = calib.optimize_daniilidis()
    result_horaud     = calib.optimize_horaud()
    result_park       = calib.optimize_park()
    result_andreff    = calib.optimize_andreff()
    result_cma_es     = calib.optimize_cma_es()
    result_cma_es_direct = calib.optimize_cma_es_direct()
    result_cma_es_fulltf = calib.optimize_cma_es_fulltf()
    
    print("#### tsai")
    calib.validate(result_tsai)
    print("#### daniilidis")
    calib.validate(result_daniilidis)
    print("#### horaud")
    calib.validate(result_horaud)
    print("#### park")
    calib.validate(result_park)
    print("#### andreff")
    calib.validate(result_andreff)
    print("#### cma_es")
    calib.validate(result_cma_es)
    print("#### cma_es_direct")
    calib.validate(result_cma_es_direct)
    print("#### cma_es_fulltf")
    calib.validate(result_cma_es_fulltf)
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     calib.br.sendTransform((t[0], t[1], t[2]), (r[0], r[1], r[2], r[3]), rospy.Time.now(), 'cam_1/camera_base', 'panda_hand')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
