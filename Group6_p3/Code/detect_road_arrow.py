"""
Road arrow detection using BEV + adaptive threshold + template matching.
"""
import cv2
import numpy as np


def get_bev_homography(K, R, t, bev_size=(400, 600), world_range=(-4, 4, 10, 40)):
    K = np.array(K); R = np.array(R); t = np.array(t)
    x1, x2, y1, y2 = world_range
    pts_world = np.float32([[x1,y1,0],[x2,y1,0],[x2,y2,0],[x1,y2,0]])
    R_inv = R.T
    pts_cam_raw = (pts_world - t) @ R_inv.T
    pts_cam = np.zeros_like(pts_cam_raw)
    pts_cam[:,0] = pts_cam_raw[:,0]
    pts_cam[:,1] = -pts_cam_raw[:,2]
    pts_cam[:,2] = pts_cam_raw[:,1]
    pts_img = []
    for i in range(4):
        if pts_cam[i,2] <= 0.1: continue
        p = K @ pts_cam[i]
        pts_img.append([p[0]/p[2], p[1]/p[2]])
    if len(pts_img) < 4: return None
    bw, bh = bev_size
    pts_dst = np.float32([[0,bh],[bw,bh],[bw,0],[0,0]])
    return cv2.getPerspectiveTransform(np.float32(pts_img), pts_dst)


def _pad_to_square(roi):
    h, w = roi.shape[:2]
    side = max(h, w)
    padded = np.zeros((side, side), np.uint8)
    padded[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = roi
    return padded


def _build_templates(sz=40):
    t_straight = np.zeros((sz,sz), np.uint8)
    cv2.rectangle(t_straight, (sz//2-4, sz//4), (sz//2+4, sz-2), 255, -1)
    cv2.fillPoly(t_straight, [np.array([[sz//2, 2], [sz//2-12, sz//4+2], [sz//2+12, sz//4+2]])], 255)

    t_right = np.zeros((sz,sz), np.uint8)
    cv2.rectangle(t_right, (4, sz//2-4), (sz*3//4, sz//2+4), 255, -1)
    cv2.fillPoly(t_right, [np.array([[sz-2, sz//2], [sz*3//4-2, sz//2-12], [sz*3//4-2, sz//2+12]])], 255)

    t_left = cv2.flip(t_right, 1)

    t_circle = np.zeros((sz,sz), np.uint8)
    cv2.circle(t_circle, (sz//2, sz//2), sz//3, 255, -1)

    return {'straight': t_straight, 'right': t_right, 'left': t_left}, t_circle


_TEMPLATES, _CIRCLE = _build_templates(40)


def _classify_contour(cnt, bev_mask):
    x, y, bw, bh = cv2.boundingRect(cnt)
    sz = 40
    mask = np.zeros(bev_mask.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    roi = mask[y:y+bh, x:x+bw]
    roi_resized = cv2.resize(_pad_to_square(roi), (sz, sz))

    scores = {}
    for name, tmpl in _TEMPLATES.items():
        scores[name] = float(cv2.matchTemplate(roi_resized, tmpl, cv2.TM_CCOEFF_NORMED).max())
    circ_score = float(cv2.matchTemplate(roi_resized, _CIRCLE, cv2.TM_CCOEFF_NORMED).max())

    best_dir = max(scores, key=scores.get)
    best_score = scores[best_dir]
    is_arrow = best_score > circ_score + 0.10 and best_score > 0.55

    return is_arrow, best_dir, best_score


def detect_road_arrows(frame, K, R, t, world_range=(-4, 4, 10, 40)):
    H = get_bev_homography(K, R, t, world_range=world_range)
    if H is None:
        return []

    bw_bev, bh_bev = 400, 600
    bev = cv2.warpPerspective(frame, H, (bw_bev, bh_bev))

    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, -8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, kernel, iterations=2)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 15000:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 3.0:
            continue
        if cw < 25 or ch < 25:
            continue
        if y == 0 or y + ch >= bh_bev - 2:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0
        if solidity > 0.90:
            continue

        is_arrow, direction, score = _classify_contour(cnt, adapt)
        if not is_arrow:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_bev = M["m10"] / M["m00"]
        cy_bev = M["m01"] / M["m00"]

        x1, x2, y1, y2 = world_range
        wx = x1 + (cx_bev / bw_bev) * (x2 - x1)
        wy = y1 + (1.0 - cy_bev / bh_bev) * (y2 - y1)

        if abs(wx) > 3.5:
            continue

        H_inv = np.linalg.inv(H)
        pts_bev = np.float32([[[cx_bev-20, cy_bev-30], [cx_bev+20, cy_bev+30]]])
        pts_img = cv2.perspectiveTransform(pts_bev, H_inv)[0]
        x_min, y_min = np.min(pts_img, axis=0)
        x_max, y_max = np.max(pts_img, axis=0)

        detections.append({
            "label": "road_arrow",
            "direction": direction,
            "confidence": float(score),
            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "world_pos": [float(wx), float(wy), 0.0]
        })

    return detections
