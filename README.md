# multi_cam_calib
![Screenshot from 2022-05-25 12-51-43](https://user-images.githubusercontent.com/12738633/170248789-c41764b2-6b11-41c0-a7d8-fbe5ed10a260.png)
### Requirements and Basic Information:

This project requires ROS (noetic), OpenCV, and MoveIt! in order to work.
The calibration procedure has been tested with the Franka Emika Panda robot at the ALR Lab.



### Baseline:
- Define a launch file containing parameters for camera and ChAruco calibration (e.g. charuco_calibration_\<camera\>.launch)
- Mount ChArUco board on the robot.
- Start the camera (ROS compatible) inside the terminal.
- Run `roslaunch multi_cam_calib  charuco_calibration_<camera>.launch`
- Run `source ./<catkin_worksspace>/devel/setup.bash` on the PC running the cameras.
- Run source `<moveit_workspace>/devel/setup.sh` on  the PC running the robot controller (Robot PC).
- Run `roslaunch panda_moveit_config franka_control.launch load_gripper:=true robot_ip:=172.16.0.2` on Robot PC.

### Record Poses:

- Set robot to **white mode** and run `rosrun multi_cam_calib record_poses.py` on Robot PC.
- Move the robot to different poses visible to the camera you want to calibrate. Make sure the board can be detected at each pose!
- After recording your desired poses, save them to a file.

### Calibrate to Base:
- Define calibration configuration inside a yaml file (e.g. 'calibration_config.yaml').
- Set Panda robot to **blue** mode and run `rosrun multi_cam_calib execute_calibration.py calibration_config.yaml`.
- Enter the filename of the saved joint positions.
- Enter ID of the desired calibration method.
- After the calibration, a calibration file with the calculated translation vector and quaternion will be saved in your current directory.

### Calibration Benchmarking:
- Define calibration configuration inside a yaml file (e.g. 'calibration_config.yaml').
- Set Panda robot to **blue** mode and run `rosrun multi_cam_calib calibration_benchmark.py calibration_config.yaml`.
- Enter the filename of the saved joint positions.
- After the calibration, a calibration file with the calculated translation vector and quaternion will be saved in your current directory.
