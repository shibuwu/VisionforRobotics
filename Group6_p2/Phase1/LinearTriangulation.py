import numpy as np


def linear_triangulation(K, C1, R1, C2, R2, pts1, pts2):
    P1 = K @ R1 @ np.hstack([np.eye(3), -C1.reshape(3, 1)])
    P2 = K @ R2 @ np.hstack([np.eye(3), -C2.reshape(3, 1)])
    X_all = []
    for i in range(pts1.shape[0]):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A = np.array([u1*P1[2]-P1[0], v1*P1[2]-P1[1],
                       u2*P2[2]-P2[0], v2*P2[2]-P2[1]])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X_all.append(X[:3] / X[3])
    return np.array(X_all)
