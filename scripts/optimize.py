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

    def __init__(self):
        self.tf_L0_EE = []
        self.tf_Cam_Calib = []
        
        self.val_tf_L0_EE = []
        self.val_tf_Cam_Calib = []

        self.transDict = {}
        self.rotDict = {}

        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

        self.tf_hand_to_board = self.trans_quat_to_mat([-0.0825, 0.0285, 0.2669], [0.7071068, -0.7071068, 0.0, 0.0])
        print(self.tf_hand_to_board)

    ########################
    ### Helper Functions ###
    ########################

    def get_transform_mat(self, from_tf, to_tf):
        """
            Getting the transforms from 'from_tf' to 'to_tf'
            returns 4x4 Matrix
        """
        
        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(5.0))
        trans, rot = self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))
        return ros2mat(trans, rot)


    def get_transform(self, from_tf, to_tf):
        """
            Getting the transforms from 'from_tf' to 'to_tf'
            returns [x,y,z][x,y,z,w]
        """
        
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
            data_dict = {"rotDict": self.rotDict, "transDict": self.transDict, "tf_Cam_Calib": self.tf_Cam_Calib.tolist()}
            yaml.dump(data_dict)


    def load_yaml(self, file_name):
        with open(file_name, 'r') as stream:
            data_dict = yaml.safe_load(stream)
        
        self.rotDict = data_dict["rotDict"]
        self.transDict = data_dict["transDict"]
        self.tf_Cam_Calib = np.array(data_dict["tf_Cam_Calib"])
        

    ########################
    ### Gather Functions ###
    ########################

    def gather(self, cam_idx):
        """
            Gathering the current Transforms
            Specifically from Robot Base to EndEffector, and from Camera to CalibBoard
        """
        self.tf_L0_EE.append(self.get_transform_mat('panda_link0', 'panda_hand'))

        trans_cam_calib_list = []
        rot_cam_calib_list = []

        for _ in range(10):
            self.myprint(".")

            from_string = "cam_%i/camera_base" % cam_idx
            to_string = "cam_%i/calib_board_small" % cam_idx
            (trans, rot) = self.get_transform(from_string, to_string)
            
            trans_cam_calib_list.append(trans)
            rot_cam_calib_list.append(rot)

        trans_cam_calib_mean = np.mean(np.array(trans_cam_calib_list), axis=0)
        rot_cam_calib_mean = np.mean(np.array(rot_cam_calib_list), axis=0)

        self.transDict.update({"tf_L0_EE": self.tf_L0_EE})

        self.tf_Cam_Calib.append(ros2mat(trans_cam_calib_mean, rot_cam_calib_mean))

    def separate_validation_set(self):
        self.tf_L0_EE, self.val_tf_L0_EE, self.tf_Cam_Calib, self.val_tf_Cam_Calib = train_test_split(self.tf_L0_EE, self.tf_Cam_Calib, test_size=0.33, random_state=42)
        self.transDict = {"tf_L0_EE": self.tf_L0_EE}

    ############################
    ### Validation Functions ###
    ############################

    def validate(self, tf_cam_L0):
        """
            Applying the estimated transformations and checking the variation on the combined transformation
        """
        trans_EE_Calib_list = []
        rot_EE_Calib_list = []
        for tf_L0_EE, tf_cam_calib in zip(self.val_tf_L0_EE, self.val_tf_Cam_Calib):
            tf_EE_Calib = np.dot(np.dot(np.linalg.inv(tf_cam_calib), tf_cam_L0), tf_L0_EE)

            T, R, _, _ = decompose(tf_EE_Calib)
            trans_EE_Calib_list.append(T)
            rot_EE_Calib_list.append(mat2quat(R))
            
        print("STD Err of Translation", np.std(trans_EE_Calib_list))
        print("Variation of Translation", np.var(trans_EE_Calib_list))
        print("STD Err of Rotation", np.std(rot_EE_Calib_list))
        print("Variation of Rotation", np.var(rot_EE_Calib_list))


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

        return result

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
        pos_list = np.zeros((len(self.transDict['tf_L0_EE']),3))
        orient_list = np.zeros((len(self.transDict['tf_L0_EE']),4))

        for i in range(len(self.transDict['tf_L0_EE'])):
            tf_L0_EE = self.tf_L0_EE[i]
            tf_cam_calib = self.tf_Cam_Calib[i]
            tf_calib_cam = np.linalg.inv(tf_cam_calib)
            tf_calib_EE = np.dot(np.dot(tf_calib_cam, tf_est_cam_L0), tf_L0_EE)
            pos_list[i,:] = tf_calib_EE[0:3,3]
            orient_list[i, :] = mat2quat(tf_calib_EE[0:3, 0:3])     #accepts full transformation matrix
        sse = np.sum(np.var(pos_list, axis=0)) #+ np.sum(np.var(orient_list, axis=0))

        return sse


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
    rospy.init_node('test', anonymous=True)
    robot = Robot()
    calib = Calibration()
    
    filename = raw_input('Enter the filename with the joint_positions: ')

    if not filename: 
        filename = 'joint_states.yaml'
        cam_joint_states_val = cam_joint_states

        for joint_positions in cam_joint_states:
            robot.set_joint_positions(joint_positions)
            robot.move()
            time.sleep(2)
            calib.gather(cam_idx)
            time.sleep(1)
        
        for joint_positions in cam_joint_states_val:
            robot.set_joint_positions(joint_positions)
            robot.move()
            time.sleep(2)
            calib.gather(cam_idx)
            time.sleep(1)

        param_fle_name = "param_file_%i.yaml" % cam_idx   
        calib.save_yaml(param_fle_name)

        calib.separate_validation_set()
        
        result_tsai       = calib.optimize_tsai()
        result_daniilidis = calib.optimize_daniilidis()
        result_horaud     = calib.optimize_horaud()
        result_park       = calib.optimize_park()
        result_andreff    = calib.optimize_andreff()
        result_cma_es     = calib.optimize_cma_es()
        
        print("tsai")
        calib.validate(result_tsai)
        print("daniilidis")
        calib.validate(result_daniilidis)
        print("horaud")
        calib.validate(result_horaud)
        print("park")
        calib.validate(result_park)
        print("andreff")
        calib.validate(result_andreff)
        print("cma_es")
        print("DEBUG - VAL")
        calib.validate(result_cma_es)
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
