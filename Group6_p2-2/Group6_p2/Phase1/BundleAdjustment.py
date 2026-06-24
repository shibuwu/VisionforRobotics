import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def _camera_to_param(C, R):
    q = Rotation.from_matrix(R).as_quat()
    return np.hstack([C, q])


def _param_to_camera(param):
    C = param[:3]
    q = param[3:]
    q = q / max(np.linalg.norm(q), 1e-12)
    R = Rotation.from_quat(q).as_matrix()
    return C, R


def bundle_adjustment(
    K,
    camera_centers_init,
    camera_rotations_init,
    points3d_init,
    obs_camera_indices,
    obs_point_indices,
    obs_2d,
    visibility_matrix=None,
    fixed_camera_indices=(0, 1),
    max_nfev=200,
    loss="huber",
    f_scale=3.0,
):
    n_cams = len(camera_centers_init)
    n_pts = points3d_init.shape[0]

    fixed_set = set(int(i) for i in fixed_camera_indices if 0 <= int(i) < n_cams)
    opt_cam_indices = [i for i in range(n_cams) if i not in fixed_set]
    cam_to_opt_idx = {cam_i: j for j, cam_i in enumerate(opt_cam_indices)}

    cam_params_init = np.array(
        [_camera_to_param(camera_centers_init[i], camera_rotations_init[i]) for i in opt_cam_indices],
        dtype=float,
    )
    x0 = np.hstack([cam_params_init.reshape(-1), points3d_init.reshape(-1)])

    obs_camera_indices = np.asarray(obs_camera_indices, dtype=int)
    obs_point_indices = np.asarray(obs_point_indices, dtype=int)
    obs_2d = np.asarray(obs_2d, dtype=float)

    fixed_cameras = {
        i: (np.asarray(camera_centers_init[i], dtype=float), np.asarray(camera_rotations_init[i], dtype=float))
        for i in fixed_set
    }

    n_cam_params = 7 * len(opt_cam_indices)

    n_obs = obs_2d.shape[0]
    obs_opt_cam = np.zeros(n_obs, dtype=int)
    for k in range(n_obs):
        ci = int(obs_camera_indices[k])
        if ci in cam_to_opt_idx:
            obs_opt_cam[k] = cam_to_opt_idx[ci]

    # group observations by camera for vectorized projection
    cam_obs_groups = {}
    for k in range(n_obs):
        ci = int(obs_camera_indices[k])
        if ci not in cam_obs_groups:
            cam_obs_groups[ci] = []
        cam_obs_groups[ci].append(k)
    for ci in cam_obs_groups:
        cam_obs_groups[ci] = np.array(cam_obs_groups[ci], dtype=int)

    def unpack(params):
        n_opt = len(opt_cam_indices)
        cam_params = params[:n_cam_params].reshape(n_opt, 7) if n_opt else np.zeros((0, 7))
        pts = params[n_cam_params:].reshape(n_pts, 3)
        return cam_params, pts

    def residuals(params):
        cam_params, pts = unpack(params)
        res = np.empty(n_obs * 2, dtype=float)

        for ci, obs_idx in cam_obs_groups.items():
            if ci in fixed_set:
                C, R = fixed_cameras[ci]
            else:
                C, R = _param_to_camera(cam_params[cam_to_opt_idx[ci]])

            pt_ids = obs_point_indices[obs_idx]
            X = pts[pt_ids]
            X_cam = (R @ (X - C).T).T
            proj = (K @ X_cam.T).T
            proj_2d = proj[:, :2] / proj[:, 2:3]
            diff = proj_2d - obs_2d[obs_idx]
            res[obs_idx * 2] = diff[:, 0]
            res[obs_idx * 2 + 1] = diff[:, 1]

        return res

    # sparse jacobian structure
    m = n_obs
    n_params = n_cam_params + 3 * n_pts
    sparsity = lil_matrix((2 * m, n_params), dtype=int)
    for k in range(m):
        cam_i = int(obs_camera_indices[k])
        pt_i = int(obs_point_indices[k])

        if cam_i in cam_to_opt_idx:
            c0 = 7 * cam_to_opt_idx[cam_i]
            sparsity[2 * k, c0:c0 + 7] = 1
            sparsity[2 * k + 1, c0:c0 + 7] = 1

        p0 = n_cam_params + 3 * pt_i
        sparsity[2 * k, p0:p0 + 3] = 1
        sparsity[2 * k + 1, p0:p0 + 3] = 1

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss=loss,
        f_scale=f_scale,
        jac_sparsity=sparsity,
        max_nfev=max_nfev,
        xtol=1e-12,
    )

    cam_params_opt, pts_opt = unpack(result.x)

    C_refined = []
    R_refined = []
    for i in range(n_cams):
        if i in fixed_set:
            C, R = fixed_cameras[i]
        else:
            C, R = _param_to_camera(cam_params_opt[cam_to_opt_idx[i]])
        C_refined.append(C)
        R_refined.append(R)

    return C_refined, R_refined, pts_opt, result
