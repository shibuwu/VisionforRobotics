import numpy as np


def essential_matrix_from_F(F, K):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ Vt
    return E
