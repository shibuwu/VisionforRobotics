import numpy as np

from LinearTriangulation import linear_triangulation


def disambiguate_camera_pose(poses, K, pts1, pts2):
    C1 = np.zeros(3)
    R1 = np.eye(3)
    best_count = 0
    best_pose = None
    best_X = None
    best_valid = None
    for C2, R2 in poses:
        X = linear_triangulation(K, C1, R1, C2, R2, pts1, pts2)
        cond1 = R1[2, :] @ (X - C1).T > 0
        cond2 = R2[2, :] @ (X - C2).T > 0
        valid = cond1 & cond2
        count = np.sum(valid)
        if count > best_count:
            best_count = count
            best_pose = (C2, R2)
            best_X = X
            best_valid = valid
    return best_pose, best_X[best_valid], best_valid
