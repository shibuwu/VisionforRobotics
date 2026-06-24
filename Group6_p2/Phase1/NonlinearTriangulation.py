import numpy as np
from scipy.optimize import least_squares


def reprojection_error_triangulation(X, P1, P2, pt1, pt2):
    X_h = np.append(X, 1)
    proj1 = P1 @ X_h
    proj1 = proj1[:2] / proj1[2]
    proj2 = P2 @ X_h
    proj2 = proj2[:2] / proj2[2]
    return np.array([pt1[0]-proj1[0], pt1[1]-proj1[1],
                     pt2[0]-proj2[0], pt2[1]-proj2[1]])


def nonlinear_triangulation(K, C1, R1, C2, R2, pts1, pts2, X0):
    P1 = K @ R1 @ np.hstack([np.eye(3), -C1.reshape(3, 1)])
    P2 = K @ R2 @ np.hstack([np.eye(3), -C2.reshape(3, 1)])
    X_refined = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        result = least_squares(reprojection_error_triangulation, X0[i],
                               args=(P1, P2, pts1[i], pts2[i]))
        X_refined[i] = result.x
    return X_refined
