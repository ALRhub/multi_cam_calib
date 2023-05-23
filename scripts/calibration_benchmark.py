#!/usr/bin/env python
from typing import Dict
import matplotlib
matplotlib.use("TkAgg")
import tf
import cma  # import CMA-ES optimization algorithm
import sys
import time
import rospy
import numpy as np
import moveit_commander
import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.affines import compose, decompose
from transforms3d.euler import mat2euler
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
# - fixing issue with loading and saving robot positions in yaml file
# 
# ####### TO DO:
# - adding IF statement to only append data to list if marker detection = success  (marker likely to be moved)


# calibation class
class Calibration():
    AVAILABLE_ALGORITHMS = {
        'Tsai-Lenz': cv2.CALIB_HAND_EYE_TSAI,
        'Park': cv2.CALIB_HAND_EYE_PARK,
        'Horaud': cv2.CALIB_HAND_EYE_HORAUD,
        'Andreff': cv2.CALIB_HAND_EYE_ANDREFF,
        'Daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    def __init__(self, cam_id, cam_base_frame, load_from_file = False):
        self.load_from_file = load_from_file

        self.tf_L0_EE = []
        self.tf_Cam_Calib = []
        
        self.val_tf_L0_EE = []
        self.val_tf_Cam_Calib = []

        if not load_from_file:
            self.listener = tf.TransformListener()
            self.br = tf.TransformBroadcaster()

        self.cam_id = cam_id
        self.cam_base_frame = cam_base_frame

        # Fixed transformation between flanch of link 7 (panda robot) and corner of ChAruco board
        self.tf_hand_to_board = self.trans_quat_to_mat([-0.082, 0.029, 0.2725], [0.7071068, 0.7071068, 0.0, 0.0])


    ########################
    ### Helper Functions ###
    ########################


    # Returns transformation matrix between two ROS transformation frames
    def get_transform_mat(self, from_tf, to_tf):
        """
            Getting the transforms from 'from_tf' to 'to_tf'
            returns 4x4 Matrix
        """
        assert self.load_from_file == False, "ROS INTERACTION NOT WORKING IF LOADING FROM FILE"

        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(5.0))
        trans, rot = self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))

        return self.ros2mat(trans, rot)


    # Get transformation between two ROS transformation frames
    def get_transform(self, from_tf, to_tf):
        """
            Getting the transforms from 'from_tf' to 'to_tf'
            returns [x,y,z][x,y,z,w]
        """
        assert self.load_from_file == False, "ROS INTERACTION NOT WORKING IF LOADING FROM FILE"

        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(10.0))

        return self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))


    # Returns tranformation matrix given a translation  and rotation vector 
    def trans_quat_to_mat(self, trans, rot):
        rot_m = quat2mat(rot)
        trans_mat = np.concatenate((rot_m, np.array(trans).reshape((3, 1))), axis=1)
        trans_mat = np.concatenate((trans_mat, np.array([0, 0, 0, 1]).reshape((1,4))), axis=0)

        return trans_mat
    

    # Returns transformation matrix fiven a translation vector and quaternion
    def ros2mat(self, trans, quat):
        return compose(trans, quat2mat([quat[3], quat[0], quat[1], quat[2]]), [1]*3)


    # Interactive print method
    def dynamicprint(self, to_print):
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


    # Save transformation from robot base to end-effector to a yaml file
    def save_yaml(self, file_name):
        with open(file_name, 'w') as stream:
            data_dict = {"tf_L0_EE": [n.tolist() for n in self.tf_L0_EE], "tf_Cam_Calib": [n.tolist() for n in self.tf_Cam_Calib]}
            yaml.dump(data_dict, stream)


    # Load transform from robot base to end-effector from a yaml file
    def load_yaml(self, file_name):
        with open(file_name, 'r') as stream:
            data_dict = yaml.safe_load(stream)
        
        self.tf_L0_EE = [np.array(tf) for tf in data_dict["tf_L0_EE"]]
        self.tf_Cam_Calib = [np.array(tf) for tf in data_dict["tf_Cam_Calib"]]
    
    # Split dataset to calibration and validation data. Validation dataset conists of 20% of available data.
    def separate_validation_set(self):
        self.tf_L0_EE, self.val_tf_L0_EE, self.tf_Cam_Calib, self.val_tf_Cam_Calib = train_test_split(self.tf_L0_EE, self.tf_Cam_Calib, test_size=0.2, random_state=42)
        

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
            self.dynamic_print(".")

            from_string = "cam_%i" % self.cam_id + self.cam_base_frame
            to_string = "cam_%i/calib_board_small" % self.cam_id
            (trans, rot) = self.get_transform(from_string, to_string)
            
            trans_cam_calib_list.append(trans)
            rot_cam_calib_list.append(rot)

        trans_cam_calib_mean = np.mean(np.array(trans_cam_calib_list), axis=0)
        rot_cam_calib_mean = np.mean(np.array(rot_cam_calib_list), axis=0)

        self.tf_Cam_Calib.append(self.ros2mat(trans_cam_calib_mean, rot_cam_calib_mean))


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

        for tf_L0_EE, tf_cam_calib in zip(self.tf_L0_EE, self.tf_Cam_Calib):
            tf_calib_cam = np.linalg.inv(tf_cam_calib)
            tf_calib_L0 = np.dot(tf_calib_cam, tf_cam_L0)
            tf_calib_EE = np.dot(tf_calib_L0, tf_L0_EE)
            tf_calib_calib = np.dot(tf_calib_EE, self.tf_hand_to_board)
        
            T, R, _, _ = decompose(tf_calib_calib)
            trans_calib_calib_list.append(T)
            rot_calib_calib_list.append(mat2euler(R))
            
        trans_error = np.linalg.norm(trans_calib_calib_list, axis=1) * 1000
        rot_error = np.linalg.norm(rot_calib_calib_list, axis=1)

        return trans_error, rot_error


    #####################
    ### Optimizations ###
    #####################

    # Default OpenCV calibration methods

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

        return np.linalg.inv(result)


    # CMA based calibration methods

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
            orient_list[i, :] = mat2quat(tf_calib_EE[0:3, 0:3])
        
        # Sum of Square Error (SSE)
        SSE = np.sum(np.var(pos_list, axis=0))

        return SSE


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
            temp = np.dot(T, Yi)

            # Sum of Square Error (SSE)
            SSE = SSE + np.sum(np.square(Xi - temp[0:3]))
        
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
        
            trans, rot, _, _ = decompose(tf_calib_calib)

            # Sum of Square Error (SSE)
            SSE += np.square(np.linalg.norm(trans))

        return np.sqrt(SSE)
    

    #####################
    ###   Plotting    ###
    #####################

    # Plotting box plots of all results and 3D plots of errors of positions in space 
    def plot_results(self, cam_id, res_trans, res_rot, labels):
        # Box plots for validation errors for all methods
        fig_bp, ax_bp = plt.subplots(2, figsize=(15, 15))
        ax_bp[0].set_title('Translation Error of Calibration')
        ax_bp[0].boxplot(res_trans)
        ax_bp[1].set_title('Rotation Error of Calibration')
        ax_bp[1].boxplot(res_rot)
        plt.setp(ax_bp, xticks=[1, 2, 3, 4, 5, 6, 7, 8], xticklabels=labels)
        fig_bp.savefig("calibration_cam" + str(cam_id) + ".png", dpi=400)

        x, y, z = [], [], []
        
        for mat  in self.val_tf_Cam_Calib:
            x.append(mat[0, -1])
            y.append(mat[1, -1])
            z.append(mat[2, -1])
        
        for mat  in self.tf_Cam_Calib:
            x.append(mat[0, -1])
            y.append(mat[1, -1])
            z.append(mat[2, -1])

        # 3D error plots for validation errors for all methods
        fig = plt.figure(figsize=(18, 9)) 
        fig.suptitle('Calibration Error for Camera ' + str(cam_id), fontsize=16)
        for idx in range(len(labels)):
            ax = fig.add_subplot(2,4,(idx+1), projection='3d')
            img = ax.scatter(x, y, z, c=res_trans[idx], s=100, cmap=plt.hot())
            fig.colorbar(img)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(labels[idx])

        fig.savefig("calibration_error_3d_cam" + str(cam_id) + ".png", dpi=900)


# Robot class using Moveit!
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


    # Move to new joint positions
    def move(self):
        print("============ Moving...")
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()


    # Set new joint positions
    def set_joint_positions(self, joints):
        self.arm.set_joint_value_target(joints)


# Main method for calipration process
def main(calib_config:Dict, plot:bool=False, publish:bool=True, save:bool=True):

    if len(sys.argv) < 2:
        print("DID NOT ENTER A CAMERA ID. Syntax: \"python optimize.py <camera_id>\"")
        return
    
    cam_id = int(sys.argv[1])
    cam_base = calib_config.get("base_frame")

    if len(sys.argv) == 3:

        in_filename = sys.argv[2]
        print("Loading files from %s" % in_filename)

        rospy.init_node('calib', anonymous=True)
        calib = Calibration(cam_id, cam_base, load_from_file=True)
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
        
        for joint_positions in joint_states[0]:
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
    
    print("Result: tsai")
    res_trans_tsai, res_rot_tsai = calib.validate(result_tsai)
    print("Result:  daniilidis")
    res_trans_daniilidis, res_rot_daniilidis = calib.validate(result_daniilidis)
    print("Result: horaud")
    res_trans_horaud, res_rot_horaud = calib.validate(result_horaud)
    print("Result: park")
    res_trans_park, res_rot_park = calib.validate(result_park)
    print("Result:  andreff")
    res_trans_andreff, res_rot_andreff = calib.validate(result_andreff)
    print("Result:  cma_es")
    res_trans_cma_es, res_rot_cma_es = calib.validate(result_cma_es)
    print("Result:  cma_es_direct")
    res_trans_cma_es_direct, res_rot_cma_es_direct = calib.validate(result_cma_es_direct)
    print("Result: cma_es_fulltf")
    res_trans_cma_es_fulltf, res_rot_cma_es_fulltf = calib.validate(result_cma_es_fulltf)

    # Plot results
    if plot:
        res_trans = [res_trans_tsai, res_trans_daniilidis, res_trans_horaud, res_trans_park, res_trans_andreff, res_trans_cma_es, res_trans_cma_es_direct, res_trans_cma_es_fulltf]
        res_rot = [res_rot_tsai, res_rot_daniilidis, res_rot_horaud, res_rot_park, res_rot_andreff, res_rot_cma_es, res_rot_cma_es_direct, res_rot_cma_es_fulltf]
        labels = ['tsai', 'daniilidis', 'horaud', 'park', 'andreff', 'cma_es', 'cma_es_direct', 'cma_es_fulltf']

        calib.plot_results(cam_id=cam_id, res_trans=res_trans, res_rot=res_rot, labels=labels)


    # Choose transformation of calibration method to save and/or publish
    print("Choose desidered calibration result")
    print("OPTIONS:")
    print("tsai : 1")
    print("daniilidis : 2")
    print("horaud : 3")
    print("park : 4")
    print("andreff : 5")
    print("cma_es : 6")
    print("cma_es_direct : 7")
    print("cma_es_fulltf : 8")

    choice = raw_input("Please enter desired ID:")
    chosen_calibration = int(choice)

    if chosen_calibration == 1:
        res2hand = result_tsai
    elif chosen_calibration ==2:
        res2hand = result_daniilidis
    elif chosen_calibration ==3:
        res2hand = result_horaud
    elif chosen_calibration ==4:
        res2hand = result_park
    elif chosen_calibration ==5:
        res2hand = result_andreff
    elif chosen_calibration ==6:
        res2hand = result_cma_es
    elif chosen_calibration ==7:
        res2hand = result_cma_es_direct
    elif chosen_calibration ==8:
        res2hand = result_cma_es_fulltf

    res2hand = result_cma_es_fulltf
    trans, rot, _, _ = decompose(np.linalg.inv(res2hand))
    quat = mat2quat(rot)

    # Save resulting transformation
    if save:
        filename = "calibration_cam_%i.yaml" % cam_id  
        with open(filename, 'w') as stream:
                data_dict = {"quaternion": quat, "t_vec": trans}
                yaml.dump(data_dict, stream)

    # Publish resulting transformation
    if publish:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            calib.br.sendTransform((trans[0], trans[1], trans[2]), (quat[1], quat[2], quat[3], quat[0]), rospy.Time.now(), "cam_" + str(cam_id) + cam_base, "panda_link0")
            

if __name__ == '__main__':
    try:
        with open('calibration_config.yaml') as f:
            calib_config = yaml.load(f, Loader=SafeLoader)
        main(calib_config, plot=False, publish=True, save=True)
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
