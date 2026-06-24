import numpy as np
import cv2
import glob
import os
from scipy.optimize import least_squares


def rodrigues_to_matrix(rvec):
    rvec = np.array(rvec, dtype=np.float64).flatten()
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K_x = np.array([[0, -k[2], k[1]],
                     [k[2], 0, -k[0]],
                     [-k[1], k[0], 0]])
    return np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * K_x


def matrix_to_rodrigues(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-12:
        return np.zeros(3)
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    k = k / (2 * np.sin(theta))
    return k * theta


def corner_sub_pix(gray, corners, win_size, max_iter=30, eps=0.001):
    half_w, half_h = win_size
    refined = corners.copy().astype(np.float32)
    h, w = gray.shape
    gray_f = gray.astype(np.float64)

    gx = np.zeros_like(gray_f)
    gy = np.zeros_like(gray_f)
    gx[:, 1:-1] = (gray_f[:, 2:] - gray_f[:, :-2]) * 0.5
    gy[1:-1, :] = (gray_f[2:, :] - gray_f[:-2, :]) * 0.5

    for idx in range(len(refined)):
        cx, cy = float(refined[idx][0][0]), float(refined[idx][0][1])

        for _ in range(max_iter):
            A = np.zeros((2, 2))
            b = np.zeros(2)

            icx, icy = int(np.round(cx)), int(np.round(cy))
            x_lo = max(1, icx - half_w)
            x_hi = min(w - 2, icx + half_w)
            y_lo = max(1, icy - half_h)
            y_hi = min(h - 2, icy + half_h)

            for iy in range(y_lo, y_hi + 1):
                for ix in range(x_lo, x_hi + 1):
                    dx = gx[iy, ix]
                    dy = gy[iy, ix]

                    A[0, 0] += dx * dx
                    A[0, 1] += dx * dy
                    A[1, 0] += dx * dy
                    A[1, 1] += dy * dy

                    b[0] += dx * dx * ix + dx * dy * iy
                    b[1] += dx * dy * ix + dy * dy * iy

            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-10:
                break

            new_x = (A[1, 1] * b[0] - A[0, 1] * b[1]) / det
            new_y = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

            shift = np.sqrt((new_x - cx) ** 2 + (new_y - cy) ** 2)
            cx, cy = new_x, new_y

            if shift < eps:
                break

        refined[idx][0][0] = cx
        refined[idx][0][1] = cy

    return refined


def get_valid_roi(K, dist_coeffs, img_size):
    w, h = img_size
    k1, k2 = dist_coeffs[0], dist_coeffs[1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def output_to_input(u_out, v_out):
        x = (u_out - cx) / fx
        y = (v_out - cy) / fy
        r2 = x ** 2 + y ** 2
        rad = 1 + k1 * r2 + k2 * r2 ** 2
        return fx * x * rad + cx, fy * y * rad + cy

    N = 500
    y_range = np.linspace(0, h - 1, N)
    x_range = np.linspace(0, w - 1, N)
    valid_x_min = np.zeros(N)
    valid_x_max = np.full(N, w - 1.0)
    valid_y_min = np.zeros(N)
    valid_y_max = np.full(N, h - 1.0)

    for i, y_out in enumerate(y_range):
        lo, hi = 0.0, cx
        for _ in range(50):
            mid = (lo + hi) / 2
            ud, vd = output_to_input(mid, y_out)
            if 0 <= ud <= w - 1 and 0 <= vd <= h - 1:
                hi = mid
            else:
                lo = mid
        valid_x_min[i] = hi

        lo, hi = cx, w - 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            ud, vd = output_to_input(mid, y_out)
            if 0 <= ud <= w - 1 and 0 <= vd <= h - 1:
                lo = mid
            else:
                hi = mid
        valid_x_max[i] = lo

    for i, x_out in enumerate(x_range):
        lo, hi = 0.0, cy
        for _ in range(50):
            mid = (lo + hi) / 2
            ud, vd = output_to_input(x_out, mid)
            if 0 <= ud <= w - 1 and 0 <= vd <= h - 1:
                hi = mid
            else:
                lo = mid
        valid_y_min[i] = hi

        lo, hi = cy, h - 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            ud, vd = output_to_input(x_out, mid)
            if 0 <= ud <= w - 1 and 0 <= vd <= h - 1:
                lo = mid
            else:
                hi = mid
        valid_y_max[i] = lo

    x1 = int(np.ceil(np.max(valid_x_min)))
    y1 = int(np.ceil(np.max(valid_y_min)))
    x2 = int(np.floor(np.min(valid_x_max)))
    y2 = int(np.floor(np.min(valid_y_max)))

    return (x1, y1, x2 - x1, y2 - y1)


def detect_corners(images_path, pattern_size=(9, 6)):
    image_files_all = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
    all_corners = []
    image_files = []
    img_shape = None

    for fname in image_files_all:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            corners_refined = corner_sub_pix(gray, corners, (11, 11))
            all_corners.append(corners_refined.reshape(-1, 2))
            image_files.append(fname)

    return all_corners, image_files, img_shape


def get_world_points(pattern_size=(9, 6), square_size=21.5):
    objp = np.zeros((pattern_size[0] * pattern_size[1], 2), dtype=np.float64)
    for i in range(pattern_size[1]):
        for j in range(pattern_size[0]):
            objp[i * pattern_size[0] + j] = [j * square_size, i * square_size]
    return objp


def compute_homography(world_pts, img_pts):
    n = world_pts.shape[0]

    mean_w = np.mean(world_pts, axis=0)
    std_w = np.std(world_pts)
    T_w = np.array([
        [np.sqrt(2) / std_w, 0, -np.sqrt(2) * mean_w[0] / std_w],
        [0, np.sqrt(2) / std_w, -np.sqrt(2) * mean_w[1] / std_w],
        [0, 0, 1]
    ])

    mean_i = np.mean(img_pts, axis=0)
    std_i = np.std(img_pts)
    T_i = np.array([
        [np.sqrt(2) / std_i, 0, -np.sqrt(2) * mean_i[0] / std_i],
        [0, np.sqrt(2) / std_i, -np.sqrt(2) * mean_i[1] / std_i],
        [0, 0, 1]
    ])

    world_h = np.hstack([world_pts, np.ones((n, 1))])
    img_h = np.hstack([img_pts, np.ones((n, 1))])
    world_n = (T_w @ world_h.T).T
    img_n = (T_i @ img_h.T).T

    L = np.zeros((2 * n, 9))
    for i in range(n):
        X, Y, W = world_n[i]
        u, v, w = img_n[i]
        L[2 * i] = [X, Y, W, 0, 0, 0, -u * X, -u * Y, -u * W]
        L[2 * i + 1] = [0, 0, 0, X, Y, W, -v * X, -v * Y, -v * W]

    _, _, Vt = np.linalg.svd(L)
    H_norm = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_i) @ H_norm @ T_w
    H = H / H[2, 2]
    return H


def v_ij(H, i, j):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])


def estimate_intrinsics(homographies):
    n = len(homographies)
    V = np.zeros((2 * n, 6))
    for k, H in enumerate(homographies):
        V[2 * k] = v_ij(H, 0, 1)
        V[2 * k + 1] = v_ij(H, 0, 0) - v_ij(H, 1, 1)

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b

    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 ** 2)
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(np.abs(lam / B11))
    beta = np.sqrt(np.abs(lam * B11 / (B11 * B22 - B12 ** 2)))
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam

    K = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]])
    return K


def estimate_extrinsics(K, H):
    K_inv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * K_inv @ h1
    r2 = lam * K_inv @ h2
    r3 = np.cross(r1, r2)
    t = lam * K_inv @ h3

    Q = np.column_stack([r1, r2, r3])
    U, S, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return R, t


def project_points(K, R, t, world_pts, k):
    k1, k2 = k
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    gamma = K[0, 1]

    n = world_pts.shape[0]
    M = np.hstack([world_pts, np.zeros((n, 1))])
    extrinsic = np.hstack([R, t.reshape(3, 1)])
    M_h = np.hstack([M, np.ones((n, 1))])
    P_cam = (extrinsic @ M_h.T).T

    x = P_cam[:, 0] / P_cam[:, 2]
    y = P_cam[:, 1] / P_cam[:, 2]

    r2 = x ** 2 + y ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2

    u = fx * (x * radial) + gamma * (y * radial) + cx
    v = fy * (y * radial) + cy
    return np.column_stack([u, v])


def pack_params(K, all_rvecs, all_tvecs, k):
    params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]] + list(k)
    for rvec, tvec in zip(all_rvecs, all_tvecs):
        params.extend(rvec)
        params.extend(tvec)
    return np.array(params)


def unpack_params(params, n_images):
    fx, fy, cx, cy, gamma = params[0:5]
    K = np.array([[fx, gamma, cx], [0, fy, cy], [0, 0, 1]])
    k = params[5:7]
    all_rvecs, all_tvecs = [], []
    idx = 7
    for _ in range(n_images):
        all_rvecs.append(params[idx:idx + 3])
        all_tvecs.append(params[idx + 3:idx + 6])
        idx += 6
    return K, all_rvecs, all_tvecs, k


def reprojection_residuals(params, n_images, all_corners, world_pts):
    K, all_rvecs, all_tvecs, k = unpack_params(params, n_images)
    residuals = []
    for i in range(n_images):
        R = rodrigues_to_matrix(all_rvecs[i])
        projected = project_points(K, R, all_tvecs[i], world_pts, k)
        residuals.append((all_corners[i] - projected).flatten())
    return np.concatenate(residuals)


def compute_reprojection_error(K, all_rvecs, all_tvecs, k, all_corners, world_pts):
    total_error, total_points = 0, 0
    for i in range(len(all_corners)):
        R = rodrigues_to_matrix(all_rvecs[i])
        projected = project_points(K, R, all_tvecs[i], world_pts, k)
        error = np.sqrt(np.sum((all_corners[i] - projected) ** 2, axis=1))
        total_error += np.sum(error)
        total_points += len(error)
    return total_error / total_points


def save_reprojection_images(image_files, all_corners, all_rvecs, all_tvecs, K, k, world_pts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, fname in enumerate(image_files):
        img = cv2.imread(fname)
        base_name = os.path.splitext(os.path.basename(fname))[0]

        R = rodrigues_to_matrix(all_rvecs[i])
        reprojected = project_points(K, R, all_tvecs[i], world_pts, k)

        for pt in all_corners[i]:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (0, 255, 0), 2)
        for pt in reprojected:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

        cv2.imwrite(os.path.join(output_dir, f'{base_name}_reprojection.jpg'), img)


def save_undistorted_images(image_files, K, k, output_dir, img_shape):
    os.makedirs(output_dir, exist_ok=True)
    k1, k2 = k
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    roi = get_valid_roi(K, dist_coeffs, img_shape)

    for i, fname in enumerate(image_files):
        img = cv2.imread(fname)
        base_name = os.path.splitext(os.path.basename(fname))[0]

        undistorted = cv2.undistort(img, K, dist_coeffs, None, K)

        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            undistorted = undistorted[y:y+rh, x:x+rw]

        cv2.imwrite(os.path.join(output_dir, f'{base_name}_undistorted.jpg'), undistorted)

# Main funt6ion that call all other functions
# Note: Each functions name is self-explanatory, and the code is organized in a way that follows the typical camera calibration pipeline.
def main():
    images_path = 'Calibration_Imgs'
    pattern_size = (9, 6)
    square_size = 21.5
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    all_corners, image_files, img_shape = detect_corners(images_path, pattern_size)
    n_images = len(all_corners)
    world_pts = get_world_points(pattern_size, square_size)

    homographies = [compute_homography(world_pts, all_corners[i]) for i in range(n_images)]

    K = estimate_intrinsics(homographies)

    all_rvecs, all_tvecs = [], []
    for i in range(n_images):
        R, t = estimate_extrinsics(K, homographies[i])
        all_rvecs.append(matrix_to_rodrigues(R))
        all_tvecs.append(t)

    k_init = [0.0, 0.0]
    init_error = compute_reprojection_error(K, all_rvecs, all_tvecs, k_init, all_corners, world_pts)

    params_init = pack_params(K, all_rvecs, all_tvecs, k_init)
    result = least_squares(reprojection_residuals, params_init,
                           args=(n_images, all_corners, world_pts), method='lm')
    K_opt, all_rvecs_opt, all_tvecs_opt, k_opt = unpack_params(result.x, n_images)
    final_error = compute_reprojection_error(K_opt, all_rvecs_opt, all_tvecs_opt, k_opt, all_corners, world_pts)

    print(f"\nK = \n{K_opt}")
    print(f"\nk = [{k_opt[0]:.6f}, {k_opt[1]:.6f}]")
    print(f"\nReprojection Error: {init_error:.4f} -> {final_error:.4f} pixels")

    save_reprojection_images(image_files, all_corners, all_rvecs_opt, all_tvecs_opt,
                             K_opt, k_opt, world_pts, os.path.join(output_dir, 'reprojections'))
    save_undistorted_images(image_files, K_opt, k_opt, os.path.join(output_dir, 'undistorted'), img_shape)


if __name__ == '__main__':
    main()