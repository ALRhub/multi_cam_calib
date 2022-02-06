import numpy as np
import tf
import cma  # import CMA-ES optimization algorithm

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


listener = tf.TransformListener()

trans_LO_marker = np.load('trans_L0_markerData.npy')
trans_camera_marker = np.load('trans_camera_markerData.npy')

# Training on the first 40 samples
trans_LO_markerTrain = trans_LO_marker#[0:40]
trans_camera_markerTrain = trans_camera_marker#[0:40]
numParameters = 7

for i in range(1, len(trans_LO_markerTrain)):
    distanceRobot = np.linalg.norm(trans_LO_markerTrain[i] - trans_LO_markerTrain[i - 1])
    distanceCamera = np.linalg.norm(trans_camera_markerTrain[i] - trans_camera_markerTrain[i - 1])

    print('Distances: Robot:{}, Camera: {}, Difference: {}'.format(distanceRobot, distanceCamera,
                                                                   distanceRobot - distanceCamera))

def myfun(x):
    trans = [x[0], x[1], x[2]]
    rot = [x[3], x[4], x[5], x[6]]
    mag = np.sqrt(np.sum(rot[i] * rot[i] for i in range(len(rot))))
    rotNormalized = rot / mag
    T = listener.fromTranslationRotation(trans, rotNormalized)

    SSE = 0

    for i in range(0, len(trans_LO_markerTrain)):
        Xi = trans_LO_markerTrain[i]
        Yi = trans_camera_markerTrain[i]
        Yi = np.append(Yi, 1)

        temp = np.dot(T, Yi)
        SSE = SSE + np.sum(np.square(Xi - temp[0:3])) # Sum of Square Error (SSE)

    return SSE

res = cma.fmin(myfun,numParameters*[0.01], 0.5)

print('Result by optimization: translation + quaternion: ')
trans = res[0][0:3]
quat = res[0][3:]
quat = quat / np.linalg.norm(quat)
print(trans, quat)

P_res = listener.fromTranslationRotation(res[0][0:3], res[0][3:7])
print('Result by optimization: transformation: ')
print(P_res)


# Testing on the last 10 unseen samples
trans_LO_markerTest = trans_LO_marker#[40:51]
trans_camera_markerTest = trans_camera_marker#[40:51]

RMSE1 = 0
MAE1 = 0
MAE2 = 0
transQuatLearned = res[0]


for i in range(0, len(trans_LO_markerTest)):
    Xi = trans_LO_markerTest[i]
    Yi = trans_camera_markerTest[i]
    Yi = np.append(Yi, 1)

    temp1 = np.dot(P_res, Yi)
    RMSE1 = RMSE1 + np.square(Xi - temp1[0:3])  # Root-mean-square error (RMSE) using P_res
    MAE1 = MAE1 + np.absolute(Xi - temp1[0:3]) # Mean Absolute Error (MAE)


# Errors on the training samples themselves
error1 = np.zeros((len(trans_LO_marker),))
error1_mat = np.zeros((len(trans_LO_marker),3))
error2 = np.zeros((len(trans_LO_marker),))

for i in range(0, len(trans_LO_marker)):
    Xi = trans_LO_marker[i]
    Yi = trans_camera_marker[i]
    Yi = np.append(Yi, 1)

    temp1 = np.dot(P_res, Yi)

    error1[i] = np.mean(np.absolute(Xi - temp1[0:3]))
    error1_mat[i, :] = np.absolute(Xi - temp1[0:3])

    # temp2 = np.dot(P_L0_ceilingcamera_mean, Yi)
    #
    # error2[i] = np.mean(np.absolute(Xi - temp2[0:3]))

RMSE1 = np.sqrt(RMSE1/len(trans_LO_markerTest))
MAE1 = MAE1/len(trans_LO_markerTest)
print('RMSE of using CMA-ES optimization: ')
print(RMSE1)
print('MAE of using CMA-ES optimization: ')
print(MAE1)



# ======== Visualization of collected points used for calibration and their corresponding error ========================
'''
# From robot-base to marker
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

x = trans_LO_marker[:,0]
y = trans_LO_marker[:,1]
z = trans_LO_marker[:,2]

ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# From camera to marker
fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')

x = trans_camera_marker[:,0]
y = trans_camera_marker[:,1]
z = trans_camera_marker[:,2]

ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# Visualization of errors of CMA-ES in combination with points from robot-base to marker: xy, xz and yz
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

x = trans_LO_marker[:,0]
y = trans_LO_marker[:,1]

ax.scatter(x, y, error1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('errors')

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')

x = trans_LO_marker[:,0]
z = trans_LO_marker[:,2]

ax.scatter(x, z, error1)
ax.set_xlabel('X Label')
ax.set_ylabel('Z Label')
ax.set_zlabel('errors')

fig = plt.figure(5)
ax = fig.add_subplot(111, projection='3d')

y = trans_LO_marker[:,1]
z = trans_LO_marker[:,2]

ax.scatter(y, z, error1)
ax.set_xlabel('Y Label')
ax.set_ylabel('Z Label')
ax.set_zlabel('errors')
'''
fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')

x = trans_LO_marker[:,0]
y = trans_LO_marker[:,1]
z = trans_LO_marker[:,2]

ax.scatter(x, y, z, s = error1 * 5000)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
