#!/usr/bin/env python3
"""
Wrapper.py - Phase 1 Panorama Stitching
"""

import os
import numpy as np
import cv2
import SingleImageStitcher as stitcher

# Parameters
MIN_MATCHES = 6
MIN_INLIERS = 5
HARRIS_K = 0.04
HARRIS_BLOCK = 2
HARRIS_APERTURE = 3
HARRIS_THRESH_RATIO = 0.001
ANMS_NBEST = 800
MAX_CANVAS_AREA = 3e7
DEBUG_VIS = True


def sort_key(name):
    stem = os.path.splitext(name)[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def extract_features(img_bgr):
    """Harris corner detection + ANMS + feature descriptors."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    # mask out black regions (panorama padding)
    valid = (gray > 0)

    dst = cv2.cornerHarris(gray_f, HARRIS_BLOCK, HARRIS_APERTURE, HARRIS_K)
    dst[~valid] = 0.0

    thresh = HARRIS_THRESH_RATIO * float(dst.max() if dst.size else 0.0)
    if thresh <= 0:
        return None, None

    y, x = np.where(dst > thresh)
    if len(x) < 10:
        return None, None

    corners = np.stack((x, y, dst[y, x]), axis=1)
    kps = stitcher.anms(corners, N_best=ANMS_NBEST)
    if kps is None or len(kps) < 10:
        return None, None

    desc, _ = stitcher.feature_descriptor(gray, kps)
    if desc is None or len(desc) < 10:
        return None, None

    return kps, desc


def check_homography(H, img_shape):
    """Basic sanity check on homography matrix."""
    if H is None:
        return False

    det = np.linalg.det(H)
    if det < 0.01 or det > 100:
        return False

    try:
        cond = np.linalg.cond(H)
        if cond > 1e8:
            return False
    except:
        return False

    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)

    max_dim = max(h, w) * 20
    if np.any(np.abs(warped) > max_dim):
        return False

    warped_flat = warped.reshape(4, 2)
    area = cv2.contourArea(warped_flat)
    original_area = h * w
    if area < original_area * 0.01 or area > original_area * 100:
        return False

    return True


def warp_and_paste(pano, img, H):
    """Warp panorama into new image frame and paste new image on top."""
    h1, w1 = pano.shape[:2]
    h2, w2 = img.shape[:2]

    det = np.linalg.det(H)
    if det < 1e-4 or det > 1e4:
        print(f"    bad homography det={det:.2e}, skipping")
        return pano

    pano_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    img_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    pano_warped = cv2.perspectiveTransform(pano_corners, H)
    all_corners = np.concatenate((pano_warped, img_corners), axis=0)

    xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

    out_w = int(xmax - xmin)
    out_h = int(ymax - ymin)

    if out_w <= 0 or out_h <= 0:
        return img

    # downscale if too big
    scale = 1.0
    area = out_w * out_h
    if area > MAX_CANVAS_AREA:
        scale = np.sqrt(MAX_CANVAS_AREA / area)
        if scale < 0.15:
            print(f"    canvas too large ({out_w}x{out_h}), skipping")
            return pano
        out_w = max(1, int(out_w * scale))
        out_h = max(1, int(out_h * scale))
        print(f"    downscaling to {out_w}x{out_h}")

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)

    result = cv2.warpPerspective(pano, S @ T @ H, (out_w, out_h))

    # paste new image
    if scale != 1.0:
        img_scaled = cv2.resize(img, (max(1, int(w2 * scale)), max(1, int(h2 * scale))),
                                interpolation=cv2.INTER_AREA)
        h2_s, w2_s = img_scaled.shape[:2]
        y1, y2 = int(ty * scale), int(ty * scale) + h2_s
        x1, x2 = int(tx * scale), int(tx * scale) + w2_s
    else:
        img_scaled = img
        h2_s, w2_s = h2, w2
        y1, y2 = ty, ty + h2
        x1, x2 = tx, tx + w2

    # clip to bounds
    if y1 < 0 or x1 < 0 or y2 > out_h or x2 > out_w:
        yy1, xx1 = max(0, y1), max(0, x1)
        yy2, xx2 = min(out_h, y2), min(out_w, x2)
        src_y1, src_x1 = yy1 - y1, xx1 - x1
        src_y2, src_x2 = src_y1 + (yy2 - yy1), src_x1 + (xx2 - xx1)
        result[yy1:yy2, xx1:xx2] = img_scaled[src_y1:src_y2, src_x1:src_x2]
    else:
        result[y1:y2, x1:x2] = img_scaled

    return result


def crop_black(img):
    """Remove black borders."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2*pad)
    h = min(img.shape[0] - y, h + 2*pad)

    return img[y:y+h, x:x+w]


def stitch_set(all_data, output_path):
    """Sequential stitching of images into panorama."""
    panorama = all_data[0][0].copy()
    stitched = False

    print(f"  Starting sequential stitch")

    for idx in range(1, len(all_data)):
        curr_img, curr_kps, curr_desc, curr_name = all_data[idx]

        # get features from current panorama
        pano_kps, pano_desc = extract_features(panorama)
        if pano_kps is None:
            print(f"    {curr_name}: no pano features")
            continue

        # match
        matches = stitcher.match_features(pano_desc, curr_desc)
        if matches is None or len(matches) < MIN_MATCHES:
            print(f"    {curr_name}: {0 if matches is None else len(matches)} matches (need {MIN_MATCHES})")
            continue

        pts_pano = np.float32([[pano_kps[m.queryIdx][0], pano_kps[m.queryIdx][1]] for m in matches])
        pts_curr = np.float32([[curr_kps[m.trainIdx][0], curr_kps[m.trainIdx][1]] for m in matches])

        # RANSAC
        H, inliers = stitcher.ransac(pts_pano, pts_curr)
        if H is None or inliers is None or len(inliers) < MIN_INLIERS:
            print(f"    {curr_name}: {0 if inliers is None else len(inliers)} inliers (need {MIN_INLIERS})")
            continue

        det = np.linalg.det(H)
        ratio = len(inliers) / len(matches)

        if det < 0.01:
            print(f"    {curr_name}: bad H (det={det:.2e})")
            continue
        if ratio < 0.3 and len(inliers) < 15:
            print(f"    {curr_name}: low inlier ratio ({ratio:.2f})")
            continue

        print(f"  Stitching {curr_name}: {len(inliers)}/{len(matches)} inliers, det={det:.2f}")

        if DEBUG_VIS:
            stitcher.visualize_matches(panorama, pano_kps, curr_img, curr_kps, matches,
                                       output_path, f"pano_vs_{curr_name}")
            stitcher.visualize_ransac(panorama, pts_pano, curr_img, pts_curr, inliers,
                                      output_path, f"pano_vs_{curr_name}")

        new_pano = warp_and_paste(panorama, curr_img, H)

        if new_pano.shape == panorama.shape and np.array_equal(new_pano, panorama):
            print(f"    {curr_name}: warp failed")
            continue

        panorama = new_pano
        stitched = True

    panorama = crop_black(panorama)
    return panorama, stitched


def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(current_folder, "../Phase1_Outputs")

    data_roots = [
        os.path.normpath(os.path.join(current_folder, "../Data/Train")),
        os.path.normpath(os.path.join(current_folder, "../Data/Test")),
    ]

    for data_root in data_roots:
        if not os.path.isdir(data_root):
            continue

        sets = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
        print(f"\nFound {len(sets)} sets in {data_root}: {sets}")

        for set_name in sets:
            print(f"\n=== {set_name} ===")

            input_path = os.path.join(data_root, set_name)
            output_path = os.path.join(output_root, set_name)
            os.makedirs(output_path, exist_ok=True)

            images = [f for f in os.listdir(input_path) if f.lower().endswith(".jpg")]
            images.sort(key=sort_key)

            if len(images) < 2:
                print(f"Not enough images, skipping")
                continue

            # extract features for all images
            all_data = []
            for img_name in images:
                img_path = os.path.join(input_path, img_name)
                name = os.path.splitext(img_name)[0]
                print(f"  Processing {img_name}...")

                img, kps, desc = stitcher.process_image_and_visualize(img_path, output_path, name)

                if img is None or kps is None or desc is None:
                    print(f"    failed")
                    continue

                all_data.append((img, kps, desc, name))

            if len(all_data) < 2:
                print(f"Not enough valid images")
                continue

            panorama, ok = stitch_set(all_data, output_path)

            if not ok:
                print(f"  No images could be stitched")

            pano_path = os.path.join(output_path, f"Panorama_{set_name}.png")
            cv2.imwrite(pano_path, panorama)
            print(f"  Saved: {pano_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
