import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _project_points(K, C, R, X_world):
    X_cam = (R @ (X_world - C).T).T
    proj = (K @ X_cam.T).T
    return proj[:, :2] / proj[:, 2:3]


def nonlinear_pnp(X_world, x_image, K, C0, R0):
    q0 = Rotation.from_matrix(R0).as_quat()
    p0 = np.hstack([C0, q0])

    def residuals(params):
        C = params[:3]
        q = params[3:]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            q = q0
        else:
            q = q / q_norm
        R = Rotation.from_quat(q).as_matrix()
        proj = _project_points(K, C, R, X_world)
        return (proj - x_image).reshape(-1)

    result = least_squares(residuals, p0, method="trf")
    p_opt = result.x
    C_opt = p_opt[:3]
    q_opt = p_opt[3:]
    q_opt = q_opt / max(np.linalg.norm(q_opt), 1e-12)
    R_opt = Rotation.from_quat(q_opt).as_matrix()
    return C_opt, R_opt
