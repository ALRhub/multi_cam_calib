#!/usr/bin/env python

from typing import Dict
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


# Main method for calipration process
def main(calib_config:Dict, plot:bool=False, publish:bool=True, save:bool=True):

    # if len(sys.argv) < 2:
    #     print("DID NOT ENTER A CAMERA ID. Syntax: \"python optimize.py <camera_id>\"")
    #     return
    
    cam_config = calib_config.get("camera")
    robot_config = calib_config.get("robot")
    cam_base = cam_config.get("base_frame")
    cam_id = cam_config.get("cam_id")
    robot_base = robot_config.get("base_frame")

    if len(sys.argv) == 2:

        in_filename = sys.argv[1]
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
