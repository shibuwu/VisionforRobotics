import numpy as np


def _project_points(K, C, R, X_world):
    X_cam = (R @ (X_world - C).T).T
    proj = (K @ X_cam.T).T
    return proj[:, :2] / proj[:, 2:3]


def linear_pnp(X_world, x_image, K):
    n = X_world.shape[0]
    x_h = np.hstack([x_image, np.ones((n, 1))])
    x_n = (np.linalg.inv(K) @ x_h.T).T
    u = x_n[:, 0]
    v = x_n[:, 1]

    X = X_world[:, 0]
    Y = X_world[:, 1]
    Z = X_world[:, 2]

    A = np.zeros((2 * n, 12))
    A[0::2, 0:4] = np.column_stack([X, Y, Z, np.ones(n)])
    A[0::2, 8:12] = -np.column_stack([u * X, u * Y, u * Z, u])
    A[1::2, 4:8] = np.column_stack([X, Y, Z, np.ones(n)])
    A[1::2, 8:12] = -np.column_stack([v * X, v * Y, v * Z, v])

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    R_tilde = P[:, :3]
    t_tilde = P[:, 3]

    U, S, Vt = np.linalg.svd(R_tilde)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        t_tilde = -t_tilde

    scale = np.mean(S)
    t = t_tilde / scale
    C = -R.T @ t
    return C, R


def pnp_reprojection_error(X_world, x_image, K, C, R):
    proj = _project_points(K, C, R, X_world)
    return np.linalg.norm(proj - x_image, axis=1)

