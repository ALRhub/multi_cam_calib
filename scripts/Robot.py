#!/usr/bin/env python

import rospy
import sys
import moveit_commander


# Robot class using Moveit!
class Robot():
    def __init__(self, robot_config):
        self.robot_config = robot_config
        print("============ Initialising...")
        print(sys.argv)
        moveit_commander.roscpp_initialize(sys.argv)
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm = moveit_commander.MoveGroupCommander(self.robot_config.get("move_group_commander"))
        self.arm.set_planner_id(self.robot_config.get("planer_id"))
        rospy.sleep(2)
        self.arm.set_end_effector_link(self.robot_config.get("tcp_frame"))    # planning wrt to panda_hand or link8
        self.arm.set_max_velocity_scaling_factor(self.robot_config.get("max_acceleration_scaling_factor"))  # scaling down velocity
        self.arm.set_max_acceleration_scaling_factor(self.robot_config.get("max_acceleration_scaling_factor"))  # scaling down velocity
        self.arm.allow_replanning(self.robot_config.get("allow_replanning"))
        self.arm.set_num_planning_attempts(self.robot_config.get("num_planning_attempts"))
        self.arm.set_goal_position_tolerance(self.robot_config.get("goal_position_tolerance"))
        self.arm.set_goal_orientation_tolerance(self.robot_config.get("goal_orientation_tolerance"))
        self.arm.set_planning_time(self.robot_config.get("planning_time"))

    
    # Moving joints to new positions 
    def move(self):
        """
            Moving joints to desired positions
        """
        print("============ Moving...")
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()

    
    # Set new joint positions 
    def set_joint_positions(self, joints):
        """
            Getting joint positions and setting them as desired targets
        """
        self.arm.set_joint_value_target(joints)