import numpy as np

from LinearPnP import linear_pnp, pnp_reprojection_error


def pnp_ransac(X_world, x_image, K, n_iters=2000, threshold=6.0):
    n = X_world.shape[0]
    best_inliers = np.array([], dtype=int)
    best_pose = None

    for _ in range(n_iters):
        idx = np.random.choice(n, size=6, replace=False)
        try:
            C_try, R_try = linear_pnp(X_world[idx], x_image[idx], K)
        except np.linalg.LinAlgError:
            continue

        err = pnp_reprojection_error(X_world, x_image, K, C_try, R_try)
        inliers = np.where(err < threshold)[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_pose = (C_try, R_try)

    # refit on all inliers
    C_best, R_best = linear_pnp(X_world[best_inliers], x_image[best_inliers], K)
    return C_best, R_best, best_inliers
