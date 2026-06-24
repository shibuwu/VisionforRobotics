import numpy as np


def estimate_fundamental_matrix(pts1, pts2):
    n = pts1.shape[0]
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    A = np.column_stack([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, np.ones(n)])
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    return F
