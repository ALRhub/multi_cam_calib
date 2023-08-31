#!/usr/bin/env python

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def transform_to_quat(rvec):
    return R.from_rotvec(rvec).as_quat()


def trans_mat_to_vecs(mtx):
    rvec = R.from_matrix(mtx[:3,:3]).as_quat()
    tvec = mtx[:3,3]

    return rvec.tolist(), tvec.tolist()


def trans_to_mat(rvec, tvec):
    rot = R.from_quat(rvec).as_matrix()
    trans = np.concatenate((rot, np.array(tvec).reshape((3,1))), axis=1)
    trans = np.concatenate((trans, np.array([0.,0.,0.,1.]).reshape((1,4))), axis=0)
    return trans


def mult_and_inv(mtxA, mtxB, mtxC):
    mtx_tr = mtxA @ mtxB @ mtxC 
    mtx_tr = np.linalg.inv(mtx_tr)

    return mtx_tr


def transform_pnt(rvecA, tvecA, rvecB, tvecB):
    rotA = R.from_rotvec(rvecA).as_matrix()
    transA = np.concatenate((rotA, np.array(tvecA).reshape((3,1))), axis=1)
    transA = np.concatenate((transA, np.array([0.,0.,0.,1.]).reshape((1,4))), axis=0)
    rotB = R.from_quat(rvecB).as_matrix()
    transB = np.concatenate((rotB, np.array(tvecB).reshape((3,1))), axis=1)
    transB = np.concatenate((transB, np.array([0.,0.,0.,1.]).reshape((1,4))), axis=0)

    tr_charuco_base = np.matmul(transB, transA)
    rvec = R.from_matrix(tr_charuco_base[:3,:3]).as_quat()
    tvec = tr_charuco_base[:3,3]

    return rvec.tolist(), tvec.tolist()


def transform_pnt_quat(rvecA, tvecA, rvecB, tvecB):
    rotA = R.from_quat(rvecA).as_matrix()
    transA = np.concatenate((rotA, np.array(tvecA).reshape((3,1))), axis=1)
    transA = np.concatenate((transA, np.array([0.,0.,0.,1.]).reshape((1,4))), axis=0)

    rotB = R.from_quat(rvecB).as_matrix()
    transB = np.concatenate((rotB, np.array(tvecB).reshape((3,1))), axis=1)
    transB = np.concatenate((transB, np.array([0.,0.,0.,1.]).reshape((1,4))), axis=0)

    tr_charuco_base = np.matmul(transB, transA)

    tvec = R.from_matrix(tr_charuco_base[:3,:3]).as_quat()
    rvec = tr_charuco_base[:3,3]

    return rvec.tolist(), tvec.tolist()


def invert_transformation(rvec, tvec):
    rot = R.from_quat(rvec).as_matrix()
    trans = np.concatenate((rot, np.array(tvec).reshape((3,1))), axis=1)
    trans = np.concatenate((trans, np.array([0.,0.,0.,1.]).reshape((1,4))), axis=0)
    trans_inv = np.linalg.inv(trans)

    rvec = R.from_matrix(trans_inv[:3,:3]).as_quat()
    tvec = trans_inv[:3,3]

    return rvec.tolist(), tvec.tolist()


if __name__ == "__main__":
    rot, trans = transform_pnt([1,2,0],[78,23,0], [1,1,1], [240,0,35])
    print("Rotation: " + str(rot))
    print("Translation:" + str(trans))
    rot_inv, trans_inv = invert_transformation(trans,rot)
    print("Rotation: " + str(rot_inv))
    print("Translation:" + str(trans_inv))