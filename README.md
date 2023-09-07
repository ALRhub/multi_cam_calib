# multi_cam_calib
![Screenshot from 2022-05-25 12-51-43](https://user-images.githubusercontent.com/12738633/170248789-c41764b2-6b11-41c0-a7d8-fbe5ed10a260.png)
## Requirements and Basic Information:

This project requires ROS (noetic), OpenCV, and MoveIt! to work. Furthermore, for this project to run, you need a ChAruco detection script publishing the ChAruco frame relative to the camera base. A usable script is provided in this repo, but you can also use your solution.
The calibration procedure has been tested with the Franka Emika Panda robot at the ALR Lab. We use two PCs, one for detecting the ChAruco board (AR PC) and one for controlling the robot (Robot PC).

The ChAruco board used during testing is printed on an aluminum composite panel and mounted with two screws on the TCP.
![board_top](https://github.com/ALRhub/multi_cam_calib/assets/12738633/e6781a8b-f83a-4534-af64-8e46d07c0261)

After mounting the board on the robot, one has to calculate the transformation from the TCP to the lower left corner of the board by hand. The configuration YAML file requires the translation vector and quaternion (see scripts/calib_config.yaml).

The pictures below show how the board is mounted on the robot:

![board_mounted_front](https://github.com/ALRhub/multi_cam_calib/assets/12738633/855d673d-3cf0-4717-9149-deb1a5dd1289) 
![board_mounted_side](https://github.com/ALRhub/multi_cam_calib/assets/12738633/9d442044-fc7f-4a5d-a280-6722812ecd30)


## Basic Tasks:
- Define a launch file containing parameters for camera and ChAruco calibration (e.g. charuco_calibration_\<camera\>.launch)
- Define calibration configuration file (YAML) containing all necessary parameters for your needs (see scripts/calib_config.yaml)
- Mount the ChArUco board on the robot.
- Start the camera (ROS compatible) inside the terminal.
- Run `roslaunch multi_cam_calib  charuco_calibration_<camera>.launch`.
  ![charuco_detect](https://github.com/ALRhub/multi_cam_calib/assets/12738633/53773f3b-3d7d-4fa8-82b7-220875641293)

   The TF tree should resemble something similar to this:
  ![charuco_tree](https://github.com/ALRhub/multi_cam_calib/assets/12738633/0465a70d-5825-4725-bebf-e878e82cc6c1)

- Run `source ./<catkin_worksspace>/devel/setup.bash` on the PC running the cameras.
- Run source `<moveit_workspace>/devel/setup.sh` on  the PC running the robot controller (Robot PC).
- Run `roslaunch panda_moveit_config franka_control.launch load_gripper:=true robot_ip:=172.16.0.2` on Robot PC.

## Record Poses:
- Set robot to interactive mode (**white mode**) and run `rosrun multi_cam_calib record_poses.py` on Robot PC.
- Move the robot to different poses visible to the camera you want to calibrate. Make sure the board can be detected at each pose!
- After recording your desired poses, save them to a file.

## Calibrate to Base:
### Execute calibration for a specific method
- Define calibration configuration inside a yaml file (e.g. 'calibration_config.yaml').
- Set Panda robot to execution mode (**blue mode**) and run `rosrun multi_cam_calib execute_calibration.py calibration_config.yaml`.
- Enter the filename of the saved joint positions.
- Enter the ID of the desired calibration method.
- After the calibration, a calibration file with the calculated translation vector and quaternion will be saved in your current directory.

## Calibration Benchmarking:
### Execute calibration for all available method
- Define calibration configuration inside a yaml file (e.g. 'calibration_config.yaml').
- Set Panda robot to **blue** mode and run `rosrun multi_cam_calib calibration_benchmark.py calibration_config.yaml`.
- Enter the filename of the saved joint positions.
- After the calibration, a calibration file with the calculated translation vector and quaternion can be saved in your current directory.

## Result:
After executing the script, the resulting translation and rotation errors are printed on the terminal. The transformations from the camera to the robot base are published to the TF tree. The frames can be visualized in RViz:
![tf_calib](https://github.com/ALRhub/multi_cam_calib/assets/12738633/ab9ee3ce-0ea2-421b-b4de-3df899a9e06d)

Besides the quantitative results printed on the terminal, the qualitative result can be visualized by adding a point cloud (if available) in RViz:
![pcl_calib](https://github.com/ALRhub/multi_cam_calib/assets/12738633/f9407854-ca61-462e-8710-371951f72cc1)

