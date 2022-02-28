#!/usr/bin/env python
import tf
import cma  # import CMA-ES optimization algorithm
import sys
import time
import rospy
import numpy as np
import moveit_commander
import matplotlib.pyplot as plt
import yaml

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
# 
# ####### TO DO:
# - adding IF statement to only append data to list if marker detection = sucess  (marker likely to be moved)
# - fixing issue with loading and saving robot positions in yaml file



class Calibrate(object):
    def __init__(self):
        super(Calibrate, self).__init__()
        self.listener = tf.TransformListener()

        self.transDict = {}
        self.rotDict = {}

        self.transDict['L0_EE'] = []
        self.transDict['calib_board_small'] = []

        self.rotDict['L0_EE'] = []
        self.rotDict['calib_board_small'] = []


        self.br = tf.TransformBroadcaster()
        self.robot = Robot()

    def find(self, index):
        (trans, rot) = self.get_transform('panda_link0', 'panda_hand')  # was panda_hand
        self.transDict['L0_EE'].append(trans)
        self.rotDict['L0_EE'].append(rot)

        self.feedback(index)

        print("recording images")

        self.camera_marker_list_trans = list()
        self.camera_marker_list_rot = list()


        for i in range(10):

            self.myprint(".")
            (trans, rot) = self.get_transform('cam_2/camera_base', 'cam2/calib_board_small')
            
            self.camera_marker_list_trans.append(trans)
            self.camera_marker_list_rot.append(rot)


        camera_marker_mean_trans1 = np.mean(np.array(self.camera_marker_list_trans1), axis=0)
        camera_marker_mean_rot1 = np.mean(np.array(self.camera_marker_list_rot1), axis=0)

        self.transDict['calib_board_small'].append(camera_marker_mean_trans1)
        self.rotDict['calib_board_small'].append(camera_marker_mean_rot1)


    def feedback(self, index):
        print('Calibration Point: ', index)

    def get_transform(self, from_tf, to_tf):
        self.listener.waitForTransform(from_tf, to_tf, rospy.Time().now(), rospy.Duration(5.0))
        return self.listener.lookupTransform(from_tf, to_tf, rospy.Time(0))

    def myprint(self, to_print):
        sys.stdout.write(to_print)
        sys.stdout.flush()

    def optimise(self):
        res = cma.fmin(self.objective_function, [0.06249777, -0.01670653 , 0.0959622, -0.54258566,  0.00218927, -0.83999462, -0.00224674], 0.2)

        testVal = self.objective_function(res[0])
        trans = res[0][0:3]
        quat = res[0][3:]
        quat = quat / np.linalg.norm(quat)
        p_res = self.listener.fromTranslationRotation(trans, quat)
        print('#########Result by optimization: ')
        print("trans=", trans)
        print("norm quat =", quat)
        return trans, quat

    def objective_function(self, x):
        trans = [x[0], x[1], x[2]]
        rot = [x[3], x[4], x[5], x[6]]
        mag = np.linalg.norm(rot)
        rot_normalized = rot / mag
        T = self.listener.fromTranslationRotation(trans, rot_normalized)

        sse = 0

        pos_list = np.zeros((len(self.transDict['L0_EE']),3))
        orient_list = np.zeros((len(self.transDict['L0_EE']),4))

        for i in range(len(self.transDict['L0_EE'])):
            trans_xi = self.transDict['L0_EE'][i]
            rot_xi = self.rotDict['L0_EE'][i]
            xi = self.listener.fromTranslationRotation(trans_xi, rot_xi)

            trans_yi = self.transDict['calib_board_small'][i]
            rot_yi = self.rotDict['calib_board_small'][i]
            yi = self.listener.fromTranslationRotation(trans_yi, rot_yi)

            temp = np.dot(T, yi)
            base_marker = np.dot(xi, temp)

            pos_i = base_marker[0:3,3]
            orient_i = tf.transformations.quaternion_from_matrix(base_marker)     #accepts full transformation matrix

            pos_list[i,:] = pos_i
            orient_list[i, :] = orient_i


        sse = np.sum(np.var(pos_list, axis=0)) #+ np.sum(np.var(orient_list, axis=0))

        return sse


class Robot(object):
    def __init__(self):
        print("============ Initialising...")
        super(Robot, self).__init__()
        self.load_config()
        moveit_commander.roscpp_initialize(sys.argv)
        self.commander = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")
        self.arm.set_planner_id("FMTkConfigDefault")
        # self.rviz_pub = rospy.Publisher('/move_group/display_planned_path', DisplayTrajectory, queue_size=10)
        rospy.sleep(2)
        self.arm.set_end_effector_link("panda_hand")    # planning wrt to panda_hand or link8
        self.arm.set_max_velocity_scaling_factor(0.15)  # scaling down velocity
        self.arm.set_max_acceleration_scaling_factor(0.15)  # scaling down velocity
        self.arm.allow_replanning(True)
        self.arm.set_num_planning_attempts(5)
        self.arm.set_goal_position_tolerance(0.005)
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_planning_time(5)


    def load_config(self):
        self.positions = list()

        # self.positions.append(
        #     [1.6658926282333477, -1.5905434331893922, -0.8023934044156755, -1.8385744939701898, -1.3999133581604277, 0.9356897540773664, 1.2820650930787836])

        self.positions.append([1.8613214036119836, -1.6045474252700807, -0.8685923647880555, -1.8556641442562853, -1.532336254954338, 1.0225641260147094, 1.22563084047607])
        self.positions.append([1.8186385031832115, -1.6377857224600656, -0.8066894561903817, -1.4770940262930734, -1.4263015176398413, 1.0411941898890904, 1.0186143840083055])
        self.positions.append([2.0117569705141443, -1.4879480900083268, -0.9372801320893424, -1.9522873871667044, -1.588093934893608, 0.9620878249577114, 1.5309687354607242])
        self.positions.append([1.5953483124864953, -1.728987421308245, -0.7260978712354388, -2.294965808732169, -1.4474761330911092, 1.185524028641837, 1.3664396503014224])
        self.positions.append([2.150329663385238, -1.44452367741721, -0.9413150320734297, -1.2862585973739624, -1.4911131376709257, 0.9479439322607858, 1.3146104096387115])
        self.positions.append([2.1475839805326293, -1.4681644755772183, -0.9252490367208208, -1.3087150056021555, -1.5916204590286527, 0.9420037919453212, 1.0932683991406644])
        #bad point#self.positions.append([2.1460148143491575, -1.3945050460270474, -0.6894294016701834, -1.3167538738250733, -1.590429002370153, 0.861458356652941, 1.1725540745982101])
        self.positions.append([2.1460424313949686, -1.3962997875213623, -0.6896415288107736, -1.3164858620507376, -1.589694334592138, 0.8589319139889309, 1.1726250531716007])
        self.positions.append([1.8145285320026534, -1.7288612552370344, -0.9594477970259531, -1.9097329351050514, -1.6331706961393357, 1.1100211820602417, 1.3260672193637917])
        self.positions.append([1.804509727586593, -1.7294354325703212, -0.9668114137649536, -1.9137320395878383, -1.632307254399572, 1.1095178586414882, 1.3264872209387166])
        self.positions.append([1.938855551285403, -1.4210409037726266, -0.7320329676355635, -1.5578145204441889, -1.58939833179542, 0.9805349189213344, 1.1916724054174763])
        self.positions.append([1.9696443782257183, -1.3236677703857422, -0.7289850858279637, -1.4331610162087849, -1.5603979936497552, 0.956717031955719, 1.1494734715436186])
        self.positions.append([1.6470321069168192, -1.7298358988080706, -0.5476876984323774, -1.690258891241891, -1.5109677527461733, 1.0635704683576312, 1.1570628444509847])
        self.positions.append([2.0260580607567515, -1.3370106499535697, -0.8379321197101048, -2.058074917793274, -1.539102213876588, 1.0482841787508557, 2.0462107854953833])
        self.positions.append([2.3404529571277757, -1.345013306753976, -0.9757109771456037, -1.503747660790171, -1.727489551288741, 0.8399246075664247, 1.3002340894809792])
        self.positions.append([2.190182910621166, -1.3906665913718088, -0.8752114057540894, -1.4508985642194747, -1.6988516413995198, 0.8390151983669826, 2.2293174041041306])
        self.positions.append([2.1402019023618526, -1.5732976630074638, -0.6574372162137713, -1.273985539163862, -1.6513633123295648, 0.8669887385368347, 1.071182748151677])
        self.positions.append([2.14304165155547, -1.4250164082390921, -0.6294460211481366, -1.4509333310808454, -1.661329243677003, 0.8382630023956299, 1.2506079856710774])
        self.positions.append([1.734694099398596, -1.1248260402679444, -0.8876182961463929, -1.4596861301149642, -1.2111662709542683, 0.9834356385639735, 1.1411745383782046])
        self.positions.append([1.7312343699250903, -1.1416948114122663, -0.9066033605166844, -1.4612980127334594, -1.2194365284953799, 1.1304229991095407, 1.1600751760048527])

    def move(self):
        print("============ Moving...")
        #print self.arm.plan()
        self.arm.go()

    def set_joint_positions(self, joints):
        self.arm.set_joint_value_target(joints)
        return self


def main():
    rospy.init_node('test', anonymous=True)
    cal = Calibrate()


    text = input("============ Do you want to recollect calibration data?")
    if text == "Y" or text == "y":
        for i, pos in enumerate(cal.robot.positions):
            cal.robot.set_joint_positions(pos).move()
            time.sleep(2)
            cal.find(i)
            time.sleep(1)

        with open('transFile.yaml', 'w') as stream:
            yaml.dump(cal.transDict, stream)

        with open('rotFile.yaml', 'w') as stream:
            yaml.dump(cal.rotDict, stream)


    else:
        with open('transFile.yaml', 'r') as stream:
            cal.transDict = yaml.load(stream)

        with open('rotFile.yaml', 'r') as stream:
            cal.rotDict = yaml.load(stream)

    (t, r) = cal.optimise()

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        cal.br.sendTransform((t[0], t[1], t[2]), (r[0], r[1], r[2], r[3]), rospy.Time.now(), 'cam_2/camera_base', 'panda_hand')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
