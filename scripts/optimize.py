#!/usr/bin/env python
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
    return compose(trans, quat2mat(quat[3], quat[0], quat[1], quat[2]), [1]*3)


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

        self.listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()

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
        
        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(5.0))
        return self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))

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
            rot_EE_L0.append(T)
            
            T, R, _, _ = decompose(tf_cam_calib)
            trans_cam_calib.append(T)
            rot_cam_calib.append(T)
            
        return (rot_EE_L0, trans_EE_L0), (rot_cam_calib, trans_cam_calib)

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
            (trans, rot) = self.get_transform(f"cam_{cam_idx}/camera_base", f"cam_{cam_idx}/calib_board_small")
            
            trans_cam_calib_list.append(trans)
            rot_cam_calib_list.append(rot)

        trans_cam_calib_mean = np.mean(np.array(trans_cam_calib_list), axis=0)
        rot_cam_calib_mean = np.mean(np.array(rot_cam_calib_list), axis=0)

        self.tf_Cam_Calib.append(ros2mat(trans_cam_calib_mean, rot_cam_calib_mean))

    def gather_val(self, cam_idx):
        """
            Gathering the current Transforms
            Specifically from Robot Base to EndEffector, and from Camera to CalibBoard
        """
        
        self.val_tf_L0_EE.append(self.get_transform_mat('panda_link0', 'panda_hand'))

        trans_cam_calib_list = []
        rot_cam_calib_list = []

        for i in range(10):
            self.myprint(".")
            (trans, rot) = self.get_transform(f"cam_{cam_idx}/camera_base", f"cam_{cam_idx}/calib_board_small")
            
            trans_cam_calib_list.append(trans)
            rot_cam_calib_list.append(rot)

        trans_cam_calib_mean = np.mean(np.array(trans_cam_calib_list), axis=0)
        rot_cam_calib_mean = np.mean(np.array(rot_cam_calib_list), axis=0)

        self.val_tf_Cam_Calib.append(ros2mat(trans_cam_calib_mean, rot_cam_calib_mean))

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
            tf_EE_Calib = np.linalg.inv(tf_cam_calib) @ tf_cam_L0 @ tf_L0_EE
            
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

    def optimise_tsai(self):
        return self._optimise_opencv('Tsai-Lenz')
    
    def optimise_daniilidis(self):
        return self._optimise_opencv('Daniilidis')
    
    def optimise_horaud(self):
        return self._optimise_opencv('Horaud')

    def optimise_park(self):
        return self._optimise_opencv('Park')

    def optimise_andreff(self):
        return self._optimise_opencv('Andreff')
    
    def _optimise_opencv(self, algorithm):
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

    def optimise_cma_es(self):
        res = cma.fmin(self.objective_function_cma_es, [0.1]*7, 0.2)

        trans = res[0][0:3]
        quat = res[0][3:]
        quat = quat / np.linalg.norm(quat)

        return compose(trans, quat, [1]*3)

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
            
            tf_calib_EE = tf_calib_cam @ tf_est_cam_L0 @ tf_L0_EE

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
        self.arm.set_num_planning_attempts(5)
        self.arm.set_goal_position_tolerance(0.005)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_planning_time(5)

    def move(self):
        print("============ Moving...")
        self.arm.go()

    def set_joint_positions(self, joints):
        self.arm.set_joint_value_target(joints)


def main():
    rospy.init_node('test', anonymous=True)
    robot = Robot()
    
    filename = input('Enter the filename with the joint_positions: ')
    if not filename: 
        filename = 'joint_states.yaml'

    with open('joint_states.yaml', 'r') as infile:
        joint_states = yaml.safe_load(infile)
    
    for cam_idx, cam_joint_states in enumerate(joint_states):
        calib = Calibration()

        for joint_positions in cam_joint_states:
            robot.set_joint_positions(joint_positions)
            time.sleep(2)
            Calibration.gather(cam_idx)
            time.sleep(1)
        
        for joint_positions in cam_joint_states_val:
            robot.set_joint_positions(joint_positions)
            time.sleep(2)
            Calibration.gather(cam_idx)
            time.sleep(1)
            
        with open(f"trans_file_{cam_idx}.yaml", 'w') as stream:
            yaml.dump(calib.transDict)

        with open(f"rot_file_{cam_idx}.yaml", 'w') as stream:
            yaml.dump(calib.rotDict)

        result_tsai = calib.optimise_tsai()
        result_daniilidis = calib.optimise_daniilidis()
        result_horaud = calib.optimise_horaud()
        result_park = calib.optimise_park()
        result_andreff = calib.optimise_andreff()
        result_cma_es = calib.optimise_cma_es()
        

    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #     calib.br.sendTransform((t[0], t[1], t[2]), (r[0], r[1], r[2], r[3]), rospy.Time.now(), 'cam_2/camera_base', 'panda_hand')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
