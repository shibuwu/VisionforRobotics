import cv2
import numpy as np
import math


def _extract_lit_region(crop_bgr, color):
    """Extract binary mask of the lit bulb from a TL crop."""
    h, w = crop_bgr.shape[:2]
    if h < 6 or w < 3:
        return None, None

    # split into thirds — top=red, mid=yellow, bot=green
    h3 = h // 3
    third_map = {"red": 0, "yellow": 1, "green": 2}
    idx = third_map.get(color, 2)

    y_start = idx * h3
    y_end = (idx + 1) * h3 if idx < 2 else h
    bulb_crop = crop_bgr[y_start:y_end, :]

    if bulb_crop.size == 0:
        return None, None

    # convert to HSV and threshold color-specific pixels
    hsv = cv2.cvtColor(bulb_crop, cv2.COLOR_BGR2HSV)

    if color == "red":
        mask1 = cv2.inRange(hsv, (0, 20, 50), (12, 255, 255))
        mask2 = cv2.inRange(hsv, (165, 20, 50), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)
    elif color == "yellow":
        mask = cv2.inRange(hsv, (15, 20, 50), (38, 255, 255))
    elif color == "green":
        mask = cv2.inRange(hsv, (35, 20, 50), (85, 255, 255))
    else:
        mask = cv2.inRange(hsv, (0, 0, 100), (180, 255, 255))

    # morphological cleanup: close small gaps, then remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask, bulb_crop


def _count_convexity_defects(cnt):
    """Count significant convexity defects in a contour.
    Arrows typically have 1-3 deep concavities, circles have 0."""
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if len(hull_indices) < 4 or len(cnt) < 4:
        return 0

    try:
        defects = cv2.convexityDefects(cnt, hull_indices)
    except cv2.error:
        return 0

    if defects is None:
        return 0

    # count defects with significant depth
    perimeter = cv2.arcLength(cnt, True)
    min_depth = perimeter * 0.03  # 3% of perimeter
    sig_defects = 0
    for d in defects:
        depth = d[0][3] / 256.0  # depth is in fixed-point
        if depth > min_depth:
            sig_defects += 1

    return sig_defects


def _analyze_shape(mask):
    """Detect arrow vs circle using circularity, solidity, defects, corners."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "none", 0.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < 15:
        return False, "none", 0.0

    # check mask brightness — dim/noisy masks give unreliable shapes
    mask_brightness = float(mask.sum()) / (mask.shape[0] * mask.shape[1] * 255)
    if mask_brightness < 0.05:  # less than 5% of pixels are lit
        return False, "none", 0.0

    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1:
        return False, "none", 0.0

    circularity = (4 * math.pi * area) / (perimeter * perimeter)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0

    # polygon approximation: circles approximate to many-sided polygons,
    # arrows to fewer sides (5-8 vertices)
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    n_vertices = len(approx)

    # convexity defects: arrows have concavities, circles don't
    n_defects = _count_convexity_defects(cnt)

    # corner detection on the mask (Harris)
    mask_float = mask.astype(np.float32)
    corners = cv2.cornerHarris(mask_float, blockSize=3, ksize=3, k=0.04)
    n_corners = np.sum(corners > 0.01 * corners.max()) if corners.max() > 0 else 0

    # scoring: multiple signals vote for arrow vs circle
    arrow_score = 0.0

    # circularity: low = more likely arrow
    if circularity < 0.45:
        arrow_score += 0.4
    elif circularity < 0.60:
        arrow_score += 0.25
    elif circularity < 0.75:
        arrow_score += 0.1

    # solidity: arrows have concavities -> lower solidity
    if solidity < 0.80:
        arrow_score += 0.25
    elif solidity < 0.90:
        arrow_score += 0.15
    elif solidity < 0.95:
        arrow_score += 0.05

    # convexity defects
    if n_defects >= 2:
        arrow_score += 0.2
    elif n_defects >= 1:
        arrow_score += 0.1

    # polygon vertices: arrows typically 5-10 vertices
    if 4 <= n_vertices <= 10:
        arrow_score += 0.1
    elif n_vertices > 12:
        arrow_score -= 0.05  # many vertices = smooth = circle-like

    # significant corners
    if n_corners > 3:
        arrow_score += 0.1

    is_arrow = arrow_score >= 0.45 and n_defects >= 1

    if not is_arrow:
        return False, "none", circularity

    direction = _get_arrow_direction(cnt, mask.shape)
    return True, direction, arrow_score


def _get_arrow_direction(contour, mask_shape):
    """PCA on contour to find arrow tip direction."""
    h, w = mask_shape[:2]

    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return "straight"

    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)

    if cov.shape != (2, 2):
        return "straight"

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    projections = centered @ principal
    pos_mask = projections > 0
    neg_mask = projections <= 0

    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return "straight"

    max_proj_idx = np.argmax(projections)
    min_proj_idx = np.argmin(projections)

    # count points near each extreme (within 20% of range)
    proj_range = projections.max() - projections.min()
    near_threshold = 0.2 * proj_range

    near_max = np.sum(projections > projections.max() - near_threshold)
    near_min = np.sum(projections < projections.min() + near_threshold)

    # the tip has FEWER points near the extreme
    if near_max <= near_min:
        head_dir = principal  # tip is in the positive direction
    else:
        head_dir = -principal  # tip is in the negative direction

    # image coords: x>0 = right, y>0 = down
    dx, dy = head_dir[0], head_dir[1]
    angle = math.degrees(math.atan2(-dy, dx))  # -dy because y is inverted

    # eigenvalue ratio: how elongated is the shape?
    ev_ratio = max(eigenvalues) / (min(eigenvalues) + 1e-6)

    if ev_ratio < 1.5:
        # nearly isotropic — direction ambiguous, default to straight
        return "straight"

    # classify angle into direction
    if -55 <= angle <= 55:
        return "right"
    elif 35 <= angle <= 145:
        return "straight"
    elif -145 <= angle <= -35:
        return "straight"  # down-arrow = treat as straight
    else:
        return "left"


def _template_match(mask, bulb_crop):
    """Fallback: match against simple arrow templates."""
    h, w = mask.shape[:2]
    if h < 8 or w < 8:
        return "none", 0.0

    template_size = (min(w, 32), min(h, 32))
    tw, th = template_size

    templates = {}

    # right arrow ▶
    t_right = np.zeros((th, tw), dtype=np.uint8)
    pts = np.array([
        [tw // 6, th // 4],
        [tw * 3 // 4, th // 2],
        [tw // 6, th * 3 // 4]
    ], dtype=np.int32)
    cv2.fillPoly(t_right, [pts], 255)
    # add tail
    cv2.rectangle(t_right, (tw // 8, th * 5 // 12),
                  (tw // 3, th * 7 // 12), 255, -1)
    templates["right"] = t_right

    # left arrow ◀
    templates["left"] = cv2.flip(t_right, 1)

    # straight (up) arrow ▲
    t_up = np.zeros((th, tw), dtype=np.uint8)
    pts = np.array([
        [tw // 2, th // 6],
        [tw * 3 // 4, th // 2],
        [tw // 4, th // 2]
    ], dtype=np.int32)
    cv2.fillPoly(t_up, [pts], 255)
    cv2.rectangle(t_up, (tw * 5 // 12, th // 2),
                  (tw * 7 // 12, th * 5 // 6), 255, -1)
    templates["straight"] = t_up

    mask_resized = cv2.resize(mask, template_size, interpolation=cv2.INTER_AREA)

    best_dir = "none"
    best_score = 0.0

    for direction, template in templates.items():
        result = cv2.matchTemplate(mask_resized, template, cv2.TM_CCOEFF_NORMED)
        score = float(result.max())
        if score > best_score:
            best_score = score
            best_dir = direction

    # circle template for comparison
    t_circle = np.zeros((th, tw), dtype=np.uint8)
    cv2.circle(t_circle, (tw // 2, th // 2), min(tw, th) // 3, 255, -1)
    result = cv2.matchTemplate(mask_resized, t_circle, cv2.TM_CCOEFF_NORMED)
    circle_score = float(result.max())

    if best_score > circle_score + 0.08 and best_score > 0.15:
        return best_dir, best_score
    return "none", circle_score


def classify_arrow(frame, bbox, color="green"):
    """Returns "none", "left", "right", or "straight" for a traffic light bbox."""
    x1, y1, x2, y2 = map(int, bbox)

    # pad bbox slightly to capture the full bulb
    pad = max(2, int((x2 - x1) * 0.05))
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w_frame, x2 + pad)
    y2 = min(h_frame, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 6:
        return "none"

    mask, bulb_crop = _extract_lit_region(crop, color)
    if mask is None or mask.sum() == 0:
        return "none"

    # shape analysis (primary)
    is_arrow, direction, shape_conf = _analyze_shape(mask)
    if is_arrow and shape_conf >= 0.45:
        return direction

    # template matching (fallback)
    tmpl_dir, tmpl_conf = _template_match(mask, bulb_crop)

    if tmpl_conf > 0.35 and tmpl_dir != "none":
        return tmpl_dir

    return "none"



if __name__ == "__main__":
    import json
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Test traffic light arrow classification")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image to test")
    parser.add_argument("--scene_dir", type=str, default=None,
                        help="Path to scene output dir (with frames/ and detections.json)")
    parser.add_argument("--out", type=str, default="arrow_test_output",
                        help="Output directory for annotated crops")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.scene_dir:
        det_path = os.path.join(args.scene_dir, "detections.json")
        frames_dir = os.path.join(args.scene_dir, "frames")

        with open(det_path) as f:
            all_frames = json.load(f)

        total_tl = 0
        arrow_count = {"none": 0, "left": 0, "right": 0, "straight": 0}

        for frame_data in all_frames:
            fidx = frame_data["frame_idx"]
            frame_path = os.path.join(frames_dir, f"frame_{fidx:05d}.jpg")
            if not os.path.exists(frame_path):
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            for det in frame_data["detections"]:
                if det["label"] != "traffic_light":
                    continue

                total_tl += 1
                color = det.get("color", "unknown")
                arrow = classify_arrow(frame, det["bbox"], color)
                arrow_count[arrow] += 1

                # save all TL crops for review, not just arrows
                x1, y1, x2, y2 = map(int, det["bbox"])
                crop = frame[y1:y2, x1:x2].copy()
                label_text = f"{arrow} ({color})"
                cv2.putText(crop, label_text,
                            (2, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 0), 1)
                out_path = os.path.join(
                    args.out, f"f{fidx}_{color}_{arrow}.jpg")
                cv2.imwrite(out_path, crop)

                if arrow != "none":
                    print(f"  Frame {fidx}: {color} TL -> arrow={arrow}")

        print(f"\nTotal TLs: {total_tl}")
        print(f"Arrow distribution: {arrow_count}")

    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Cannot read {args.image}")
            exit(1)

        h, w = frame.shape[:2]
        arrow = classify_arrow(frame, [0, 0, w, h], "green")
        print(f"Arrow: {arrow}")
    else:
        print("Provide --scene_dir or --image")
