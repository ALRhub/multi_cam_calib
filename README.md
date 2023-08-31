# multi_cam_calib

![Screenshot from 2022-05-25 12-51-43](https://user-images.githubusercontent.com/12738633/170248964-51aaea89-ebb4-4aac-878b-bb27d9992016.png)


### Baseline:

- Mount ChArUco board on robot.
- Run `source ./catkin_ws/devel/setup.bash` on AR PC.
- Run `roslaunch hl2_teleop onbase_handeye_calib.launch` on AR PC.
    
    (Make sure launch file is configured for the correct camera!)
    
- Run source `ws_moveit/devel/setup.sh` on Robot PC.
- Run `roslaunch panda_moveit_config panda_control_moveit_rviz.launch load_gripper:=true robot_ip:=172.16.0.2` on Robot PC.

### Record Poses:

- Set robot to **white mode** and run `rosrun multi_cam_calib record_poses.py` on Robot PC.
- Move the robot to different poses visible to the camera you want to calibrate. Make sure the board can be detected at each pose.
- After recording your desired poses, save them to a file.

### Calibrate to Base:

- Set robot to **blue** mode and run `rosrun multi_cam_calib optimize.py <cam_id>`
- Enter the filename of the saved joint positions.
- After the calibration is complete, a calibration file with the calculated translation vector and quaternion will be saved in `ws_moveit/`.
