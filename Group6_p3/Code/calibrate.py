import cv2
import numpy as np
import glob
import json
import os

CALIB_DIR = "P3Data/Calib"
CAMERAS = ["front", "back", "left", "right"]
CHECKERBOARD = (9, 6)  # inner corners
SQUARE_SIZE = 1.0  # unknown real size, use 1.0 (relative units)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D points in world coordinates
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

results = {}

for cam in CAMERAS:
    cam_dir = os.path.join(CALIB_DIR, cam)
    images = sorted(glob.glob(os.path.join(cam_dir, "frame*.jpg")))
    # skip undistorted subfolder images
    images = [img for img in images if "undistorted" not in img]

    objpoints = []
    imgpoints = []
    img_shape = None

    print(f"\n=== {cam} camera: {len(images)} images ===")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (w, h)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"  Found corners: {os.path.basename(fname)}")
        else:
            print(f"  MISSED corners: {os.path.basename(fname)}")

    if len(objpoints) < 3:
        print(f"  WARNING: only {len(objpoints)} images with corners, calibration may be poor")

    if len(objpoints) == 0:
        print(f"  SKIPPING {cam} - no corners found")
        continue

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print(f"  RMS reprojection error: {ret:.4f}")
    print(f"  Intrinsic matrix K:\n{K}")
    print(f"  Distortion coeffs: {dist.ravel()}")

    results[cam] = {
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "img_size": list(img_shape),
        "rms_error": ret,
        "num_images_used": len(objpoints),
    }

# Save results
out_path = os.path.join(CALIB_DIR, "calibration_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved calibration to {out_path}")
