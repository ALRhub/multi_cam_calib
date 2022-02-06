import rospy
import tf
import geometry_msgs.msg
import numpy as np
import tf2_ros

#import cma  #import CMA-ES optimization algorithm
from termios import tcflush, TCIFLUSH
import time,sys

rospy.init_node('test')
listener = tf.TransformListener()

P_L0_EE_transf = []      #Transformation from robot base (Link0) to robot end-effector
P_cam_marker_transf = [] #Transformation from camera to marker
P_calibration = []      #Transformation from robot base (Link0) to camera
trans_L0_marker_list = []    # Translation form robot base to marker
trans_camera_marker_list = []
trans_camera_marker_std_list = []


# Robot End-effector to marker transformation
quat = tf.transformations.quaternion_from_euler(0, 0, np.pi / 4)

trans_EE_marker = np.array([0.0, 0.0, 0.0063])    #using average of 2 markers

P_EE_marker = listener.fromTranslationRotation(trans_EE_marker, quat)

N_samples = 20  # Number of samples to capture
N_samplesCamera2Marker = 25 # Number of samples to capture from camera to marker and then average them
for i in range(0,N_samples):

    sys.stdin.flush()
    rospy.sleep(1)
    print('Press enter to continue: ')
    input = sys.stdin.readline()


    # Robot base to End-effector transformation
    listener.waitForTransform("panda_link0", "panda_EE",  rospy.Time().now(), rospy.Duration(10.0))
    (trans,rot) = listener.lookupTransform('panda_link0', 'panda_EE', rospy.Time(0))
    P_L0_EE = listener.fromTranslationRotation(trans, rot)

    P_L0_EE_transf.append(P_L0_EE)

    # Translation from robot base to marker
    P_L0_marker = P_L0_EE.dot(P_EE_marker)
    trans_L0_marker = P_L0_marker[0:3,3]
    trans_L0_marker_list.append(trans_L0_marker)

    # Camera to marker transformation
    P_ceilingcamera_marker_list = []
    for j in range(0,N_samplesCamera2Marker):
        sys.stdout.write('.')
        sys.stdout.flush()
        listener.waitForTransform("cam_2/camera_base", "cam2/calib_board_small", rospy.Time().now(), rospy.Duration(10.0))
        (trans,rot) = listener.lookupTransform('cam_2/camera_base', 'cam2/calib_board_small', rospy.Time(0))

        P_ceilingcamera_marker = listener.fromTranslationRotation(trans, rot)
        P_ceilingcamera_marker_list.append(P_ceilingcamera_marker)


    P_ceilingcamera_marker_mean = np.mean(np.array(P_ceilingcamera_marker_list), axis = 0)
    P_ceilingcamera_marker_std = np.std(np.array(P_ceilingcamera_marker_list), axis=0)

    P_cam_marker_transf.append(P_ceilingcamera_marker_mean)
    trans_camera_marker_list.append(P_ceilingcamera_marker_mean[0:3,3])
    trans_camera_marker_std_list.append(P_ceilingcamera_marker_std[0:3, 3])


    print('Calibration Point: '+ str(i+1) + '/' + str(N_samples))
    print('Mean Translation:')
    print(trans_camera_marker_list[-1])
    print('Robot Translation:')
    print(trans_L0_marker_list[-1])

    print('Std Translation:')
    print(trans_camera_marker_std_list[-1])

    if i > 0:
        distanceRobot = np.linalg.norm(trans_L0_marker_list[-1] - trans_L0_marker_list[-2])
        distanceCamera = np.linalg.norm(trans_camera_marker_list[-1] - trans_camera_marker_list[-2])

        print('Distances: Robot:{}, Camera: {}, Difference: {}'.format(distanceRobot, distanceCamera, distanceRobot - distanceCamera))


P_L0_ceilingcamera_mean = np.mean(np.array(P_calibration), axis = 0)
np.save('P_calibrationCeilingData', P_calibration)
print('Mean of ' + str(N_samples) + ' RobotBase - Camera transformations')
print(P_L0_ceilingcamera_mean)

# Saving data for optimization
np.save('trans_L0_markerData',trans_L0_marker_list)
np.save('trans_camera_markerData',trans_camera_marker_list)
np.save('trans_camera_markerData_std',trans_camera_marker_std_list)

