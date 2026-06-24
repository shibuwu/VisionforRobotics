import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def anms(corner_list, N_best=100):
    N_strong = corner_list.shape[0]
    sorted_indices = np.argsort(corner_list[:, 2])[::-1]
    sorted_corners = corner_list[sorted_indices]

    r = np.inf * np.ones((N_strong, 1))
    limit = min(N_strong, 2000)

    for i in range(limit):
        for j in range(0, i):
            x_i, y_i = sorted_corners[i, 0], sorted_corners[i, 1]
            x_j, y_j = sorted_corners[j, 0], sorted_corners[j, 1]

            if sorted_corners[j, 2] > 0.9 * sorted_corners[i, 2]:
                dist = (x_i - x_j)**2 + (y_i - y_j)**2
                if dist < r[i]:
                    r[i] = dist

    best_indices = np.argsort(r[:, 0])[::-1]
    final_corners = sorted_corners[best_indices[:N_best]]
    return final_corners


def feature_descriptor(img, corners):
    descriptors = []
    patch_list = []
    pad = 20

    img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    for point in corners:
        x, y = int(point[0]), int(point[1])
        x_pad, y_pad = x + pad, y + pad

    
        patch = img_padded[y_pad-20:y_pad+21, x_pad-20:x_pad+21]
        if patch.shape != (41, 41):
            continue

        blurred = cv2.GaussianBlur(patch, (5, 5), 0)
        subsampled = blurred[::5, ::5]  # 9x9
        patch_list.append(subsampled)

        vector = subsampled[:8, :8].reshape(-1, 1).astype(np.float32)
        mean = float(vector.mean())
        std = float(vector.std())
        if std < 1e-5:
            std = 1.0
        vector = (vector - mean) / std
        descriptors.append(vector)

    if len(descriptors) == 0:
        return np.zeros((0, 64, 1), dtype=np.float32), patch_list

    return np.array(descriptors, dtype=np.float32), patch_list


def match_features(desc1, desc2, ratio=0.8):
    """Match features using ratio test."""
    matches = []
    N1, N2 = len(desc1), len(desc2)

    if N1 == 0 or N2 < 2:
        return matches

    ratio2 = ratio * ratio

    for i in range(N1):
        diffs = []
        for j in range(N2):
            dist = np.sum((desc1[i] - desc2[j]) ** 2)
            diffs.append(dist)

        diffs = np.array(diffs)
        sorted_indices = np.argsort(diffs)

        best_idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1])

        if diffs[best_idx] < ratio2 * diffs[second_idx]:
            matches.append(cv2.DMatch(i, best_idx, float(diffs[best_idx])))

    return matches


def ransac(pts1, pts2, n_iter=10000, thresh=4.0):
    """RANSAC homography estimation."""
    if len(pts1) < 4:
        return None, []

    best_inliers = []
    n_pts = len(pts1)

    for _ in range(n_iter):
        idx = np.random.choice(n_pts, 4, replace=False)
        src = pts1[idx].astype(np.float32)
        dst = pts2[idx].astype(np.float32)

        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            continue

        inliers = []
        for i in range(n_pts):
            p1 = np.array([pts1[i][0], pts1[i][1], 1])
            p2_est = H @ p1
            if abs(p2_est[2]) < 1e-10:
                continue
            p2_est = p2_est / p2_est[2]

            err = np.sqrt((p2_est[0] - pts2[i][0])**2 + (p2_est[1] - pts2[i][1])**2)
            if err < thresh:
                inliers.append(i)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

        if len(best_inliers) > 0.9 * n_pts:
            break

    if len(best_inliers) < 4:
        return None, []

    src_inliers = pts1[best_inliers].astype(np.float32)
    dst_inliers = pts2[best_inliers].astype(np.float32)
    H_final, _ = cv2.findHomography(src_inliers, dst_inliers, 0)

    return H_final, best_inliers


def process_image_and_visualize(img_path, output_dir, file_id, N_best=500):
    """Load image, detect corners, run ANMS, compute descriptors, save visualizations."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_path}")
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)

    thresh = 0.001 * dst.max()
    y, x = np.where(dst > thresh)
    corners = np.stack((x, y, dst[y, x]), axis=1)

    # save corner detection
    vis_harris = img.copy()
    for p in corners:
        cv2.circle(vis_harris, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(output_dir, f'{file_id}_corner.png'), vis_harris)

    # ANMS
    best_corners = anms(corners, N_best=N_best)

    vis_anms = img.copy()
    for p in best_corners:
        cv2.circle(vis_anms, (int(p[0]), int(p[1])), 4, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(output_dir, f'{file_id}_anms.png'), vis_anms)

    # descriptors
    descriptors, patches = feature_descriptor(gray, best_corners)

    # plot descriptors and patches
    if len(descriptors) > 0 and len(patches) > 0:
        n_show = min(3, len(descriptors), len(patches))

        fig, axes = plt.subplots(2, n_show, figsize=(8, 6))
        if n_show == 1:
            axes = axes.reshape(2, 1)

        for i in range(n_show):
            # top row: 1D descriptor plot
            axes[0, i].plot(descriptors[i].flatten())
            axes[0, i].set_title(f'Feature {i+1}')
            axes[0, i].set_xlabel('Index')
            axes[0, i].set_ylabel('Intensity')
            axes[0, i].grid(True)

            # bottom row: patch visualization
            axes[1, i].imshow(patches[i], cmap='gray')
            axes[1, i].set_title(f'Patch {i+1}')
            axes[1, i].axis('off')

        plt.suptitle(f'Feature Descriptors - {file_id}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file_id}_FD.png'))
        plt.close()

    return img, best_corners, descriptors


def visualize_matches(img1, kps1, img2, kps2, matches, output_dir, name_pair):
    """Draw feature matches between two images."""
    kp_obj1 = [cv2.KeyPoint(p[0], p[1], 10) for p in kps1]
    kp_obj2 = [cv2.KeyPoint(p[0], p[1], 10) for p in kps2]

    match_img = cv2.drawMatches(img1, kp_obj1, img2, kp_obj2, matches, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(output_dir, f'Matching_{name_pair}.png'), match_img)


def visualize_ransac(img1, pts1, img2, pts2, inliers, output_dir, name_pair):
    """Draw inlier matches after RANSAC."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1:] = img2

    for i in inliers:
        p1 = (int(pts1[i][0]), int(pts1[i][1]))
        p2 = (int(pts2[i][0]) + w1, int(pts2[i][1]))
        cv2.circle(out, p1, 4, (0, 255, 0), -1)
        cv2.circle(out, p2, 4, (0, 255, 0), -1)
        cv2.line(out, p1, p2, (0, 255, 255), 1)

    cv2.imwrite(os.path.join(output_dir, f'RANSAC_{name_pair}.png'), out)


