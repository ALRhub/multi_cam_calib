#!/usr/bin/env python

import rospy
import moveit_commander
import sys
import yaml
from yaml.loader import SafeLoader

# Method for recording new robot joint positions
def main(calib_config):
    robot_config = calib_config.get("robot")
    rospy.init_node('pose_rec', anonymous=True)

    print("This tool will allow you to record poses for the robot into a csv file")
    print("ONLY USE THIS IN THE WHITE MODE")
    print("HELP:")
    print("'s' to store the current pose")
    print("'c' to add a as new list (for example between camera angles)")
    print("'q' to finish and save the stored poses including comment lines")
    
    
    ## Initializing the Robot ##
    moveit_commander.roscpp_initialize(sys.argv)
    arm = moveit_commander.MoveGroupCommander(robot_config.get("move_group_commander"))
    arm.set_planner_id(robot_config.get("planer_id"))
    arm.set_end_effector_link(robot_config.get("tcp_frame"))    # planning wrt to robot tcp
    
    joint_states = [[]]
    while True:
        
        text = raw_input()
        
        if text == "q":
            # TODO safe the poses
            file_name = raw_input('Enter a filename, (if empty "joint_states.yaml")')
            if not file_name:
                file_name = "joint_states.yaml"
                
            with open(file_name, "w") as outfile:
                yaml.dump(joint_states, outfile)
            break
                
        elif text == "s":
            joint_states[-1].append(arm.get_current_joint_values())
            print("Recorded ", joint_states[-1][-1])
        elif text == "c":
            joint_states.append([])
    

if __name__ == '__main__':
    try:
        with open('calibration_config.yaml') as f:
            calib_config = yaml.load(f, Loader=SafeLoader)
        main(calib_config)
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
