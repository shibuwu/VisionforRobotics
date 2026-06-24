import numpy as np


def extract_camera_pose(E):
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    poses = []
    for C, R in [(U[:, 2], U@W@Vt), (-U[:, 2], U@W@Vt),
                 (U[:, 2], U@W.T@Vt), (-U[:, 2], U@W.T@Vt)]:
        if np.linalg.det(R) < 0:
            C, R = -C, -R
        poses.append((C, R))
    return poses
