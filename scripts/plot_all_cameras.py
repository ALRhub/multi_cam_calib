import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D


def main():
    res_trans = []
    positions = []

    for i in range(1,5):
        filename = 'trans_errors_cam'+str(i)
        with open(filename, 'rb') as f:
            res_trans.append((np.load(f).tolist()))
        
        filename = 'pos_cam'+str(i)
        with open(filename, 'rb') as f:
            positions.append(np.load(f).tolist())

    tmp_err = []    
    for (l1, l2, l3, l4) in zip(res_trans[0], res_trans[1], res_trans[2], res_trans[3]):
        tmp_err.append(l1+l2+l3+l4)

    tmp_pos = []
    for (l1, l2, l3, l4) in zip(positions[0], positions[1], positions[2], positions[3]):
        tmp_pos.append(l1+l2+l3+l4)
    
    x = tmp_pos[0]
    y = tmp_pos[1]
    z = tmp_pos[2]

    res_trans = tmp_err
    labels = ['tsai', 'daniilidis', 'horaud', 'park', 'andreff', 'cma_es', 'cma_es_direct', 'cma_es_fulltf']

    fig_bp, ax_bp = plt.subplots(1, figsize=(15, 15))
    ax_bp.set_title('Translation Error of Calibration')
    ax_bp.boxplot(res_trans)
    
    plt.setp(ax_bp, xticks=[1, 2, 3, 4, 5, 6, 7, 8], xticklabels=labels)
    fig_bp.savefig("calibration_all_cams.png", dpi=400)

    fig = plt.figure(figsize=(18, 9)) 
    fig.suptitle('Calibration Error for all Cameras ', fontsize=16)
    for idx in range(len(labels)):
        ax = fig.add_subplot(2,4,(idx+1), projection='3d')
        img = ax.scatter(x, y, z, c=res_trans[idx], s=100, cmap=plt.hot())
        fig.colorbar(img)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(labels[idx])

    fig.savefig("calibration_error_3d_cam_all.png", dpi=900)
    plt.show()

if __name__ == '__main__':
    main()