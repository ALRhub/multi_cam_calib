#!/usr/bin/env python

import rospy
import moveit_commander
import sys
import yaml

def main():
    rospy.init_node('test', anonymous=True)
    print("This tool will allow you to record poses for the robot into a csv file")
    print("ONLY USE THIS IN THE WHITE MODE")
    print("HELP:")
    print("'s' to store the current pose")
    print("'c' to add a as new list (for example between camera angles)")
    print("'q' to finish and save the stored poses including comment lines")
    
    
    ## Initializing the Robot ##
    moveit_commander.roscpp_initialize(sys.argv)
    arm = moveit_commander.MoveGroupCommander("panda_arm")
    arm.set_planner_id("FMTkConfigDefault")
    arm.set_end_effector_link("panda_hand")    # planning wrt to panda_hand or link8
    
    joint_states = [[]]
    while True:
        
        text = raw_input()
        
        if text == "q":
            # TODO safe the poses
            filename = raw_input('Enter a filename, (if empty "joint_states.yaml")')
            if not filename:
                filename = "joint_states.yaml"
                
            with open(filename, "w") as outfile:
                yaml.dump(joint_states, outfile)
            break
                
        elif text == "s":
            joint_states[-1].append(arm.get_current_joint_values())
            print("Recorded ", joint_states[-1][-1])
        elif text == "c":
            joint_states.append([])
    

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)