import numpy as np

from EstimateFundamentalMatrix import estimate_fundamental_matrix


def get_inliers_ransac(pts1, pts2, n_iters=2000, threshold=0.05):
    n = pts1.shape[0]
    best_inliers = []
    ones = np.ones((n, 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    for _ in range(n_iters):
        idx = np.random.choice(n, 8, replace=False)
        F = estimate_fundamental_matrix(pts1[idx], pts2[idx])
        x2Fx1 = np.sum(pts2_h * (F @ pts1_h.T).T, axis=1)
        inliers = np.where(np.abs(x2Fx1) < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    F_best = estimate_fundamental_matrix(pts1[best_inliers], pts2[best_inliers])
    return F_best, best_inliers
