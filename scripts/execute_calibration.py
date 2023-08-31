#!/usr/bin/env python


import os
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
from Calibration import Calibration
from Robot import Robot

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


# Main method of calibration process
def main(calib_config, plot=False, publish=True, save=True):

    # if len(sys.argv) < 2:
    #     print("DID NOT ENTER A CAMERA ID. Syntax: \"python optimize.py <camera_id>\"")
    #     return
    
    cam_config = calib_config.get("camera")
    robot_config = calib_config.get("robot")
    print(robot_config)
    cam_base = cam_config.get("base_frame")
    cam_id = cam_config.get("cam_id") #int(sys.argv[1])
    robot_base = robot_config.get("base_frame")

    if len(sys.argv) == 3:
        in_filename = sys.argv[2]
        print("Loading files from %s" % in_filename)

        rospy.init_node('calib', anonymous=True)
        
        calib = Calibration(cam_id, cam_base, calib_config, load_from_file=True)
        calib.load_yaml(in_filename)
        
    else:
        rospy.init_node('test', anonymous=True)
        robot = Robot(robot_config)
        calib = Calibration(cam_id, cam_base, calib_config)

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
    
    print("Available calibration solvers")
    print("OPTIONS:")
    print("tsai : 1")
    print("daniilidis : 2")
    print("horaud : 3")
    print("park : 4")
    print("andreff : 5")
    print("cma_es : 6")
    print("cma_es_direct : 7")
    print("cma_es_fulltf : 8")

    
    choice = raw_input("Please enter desired solver ID:")
    chosen_calibration = int(choice)
    
    if chosen_calibration == 1:
        result_tsai = calib.optimize_tsai()
        res_trans, res_rot = calib.validate(result_tsai)
        label = ['tsai']
        res2hand = result_tsai
    elif chosen_calibration == 2:
        result_daniilidis = calib.optimize_daniilidis()
        res_trans, res_rot = calib.validate(result_daniilidis)
        label = ['daniilidis']
        res2hand = result_daniilidis
    elif chosen_calibration == 3:
        result_horaud = calib.optimize_horaud()
        res_trans, res_rot = calib.validate(result_horaud)
        label = ['horaud']
        res2hand = result_horaud
    elif chosen_calibration == 4:
        result_park = calib.optimize_park()
        res_trans, res_rot = calib.validate(result_park)
        label = ['park']
        res2hand = result_park
    elif chosen_calibration == 5:
        result_andreff = calib.optimize_andreff()
        res_trans, res_rot = calib.validate(result_andreff)
        label = ['andreff']
        res2hand = result_andreff
    elif chosen_calibration == 6:
        result_cma_es = calib.optimize_cma_es()
        res_trans, res_rot = calib.validate(result_cma_es)
        label = ['cma_es']
        res2hand = result_cma_es
    elif chosen_calibration == 7:
        result_cma_es_direct = calib.optimize_cma_es_direct()
        res_trans, res_rot = calib.validate(result_cma_es_direct)
        label = ['cma_es_direct']
        res2hand = result_cma_es_direct
    elif chosen_calibration == 8:
        result_cma_es_fulltf = calib.optimize_cma_es_fulltf()
        res_trans, res_rot = calib.validate(result_cma_es_fulltf)
        label = ['cma_es_fulltf']
        res2hand = result_cma_es_fulltf
    else:
        print("{} is not a valid option!".format(chosen_calibration))
    

    # Plot results
    if plot:
        calib.plot_results(cam_id=cam_id, res_trans=res_trans, res_rot=res_rot, labels=label)


    res2hand = result_cma_es_fulltf
    trans, rot, _, _ = decompose(np.linalg.inv(res2hand))
    quat = mat2quat(rot)

    # Save results transformation
    if save:
        save_path = calib_config.get("save_path")
        if os.path.exists(save_path):
            file_name = os.path.join(save_path, "calibration_cam_%i.yaml" % cam_id)
        else:
            file_name = "calibration_cam_%i.yaml" % cam_id  
        
        with open(file_name, 'w') as stream:
                data_dict = {"quaternion": quat, "translation": trans}
                yaml.dump(data_dict, stream)

    # Publish resulting transformation
    if publish:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            calib.br.sendTransform((trans[0], trans[1], trans[2]), (quat[1], quat[2], quat[3], quat[0]), rospy.Time.now(), cam_base, robot_base)
            

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("DID NOT ENTER A CAMERA ID. Syntax: \"python optimize.py <config_file.yaml>\"")
        exit()
    try:
        with open(sys.argv[1]) as f:
            calib_config = yaml.load(f, Loader=SafeLoader)

        main(calib_config, plot=calib_config.get("plot"), publish=calib_config.get("publish"), save=calib_config.get("save"))
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
