"""
Speed limit sign detection (Phase 2).

Two-stage approach:
  1. Candidate detection via YOLO-World (open-vocab "road sign")
  2. OCR via EasyOCR to read the speed limit number
"""

import cv2

VALID_SPEEDS = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75}

_yolo_world_model = None


def _get_yolo_world():
    global _yolo_world_model
    if _yolo_world_model is None:
        from ultralytics import YOLO
        _yolo_world_model = YOLO("yolov8s-worldv2.pt")
        _yolo_world_model.set_classes(["road sign"])
    return _yolo_world_model


def init_ocr_reader():
    """Initialize EasyOCR reader. Call once and reuse."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        return reader
    except ImportError:
        print("WARNING: easyocr not installed. Run: pip install easyocr")
        return None


def _find_sign_candidates(frame):
    """Find speed sign candidates using YOLO-World open-vocab detection."""
    model = _get_yolo_world()
    results = model(frame, conf=0.10, verbose=False)[0]

    candidates = []
    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        bh = y2 - y1
        bw = x2 - x1
        # basic size filter: not too tiny, not huge
        if bw < 15 or bh < 20:
            continue
        # signs should be in upper 70% of frame
        if y1 > frame.shape[0] * 0.7:
            continue
        candidates.append((x1, y1, x2, y2))

    return candidates


def _ocr_sign(frame_crop, reader):
    """Run OCR on a speed sign candidate crop and extract the speed number."""
    if reader is None:
        return None, 0.0

    h, w = frame_crop.shape[:2]
    if h < 60 or w < 50:
        scale = max(60 / h, 50 / w, 1.0)
        frame_crop = cv2.resize(frame_crop, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    results_orig = reader.readtext(frame_crop, detail=1, paragraph=False,
                                    allowlist='0123456789 SPEEDLIMIT')
    results_bin = reader.readtext(binary, detail=1, paragraph=False,
                                   allowlist='0123456789 SPEEDLIMIT')

    all_results = results_orig + results_bin

    best_speed = None
    best_conf = 0.0

    for (bbox_pts, text, conf) in all_results:
        text = text.strip().upper()
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            continue
        try:
            num = int(digits)
        except ValueError:
            continue
        if num in VALID_SPEEDS and conf > best_conf:
            best_speed = num
            best_conf = conf

    return best_speed, best_conf


def detect_speed_signs(frame, reader, min_conf=0.45):
    """Detect speed limit signs in a frame."""
    if reader is None:
        return []

    candidates = _find_sign_candidates(frame)
    detections = []

    for (x1, y1, x2, y2) in candidates:
        pad_x = max(3, int((x2 - x1) * 0.1))
        pad_y = max(3, int((y2 - y1) * 0.1))
        h, w = frame.shape[:2]
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        speed, conf = _ocr_sign(crop, reader)

        if speed is not None and conf >= min_conf:
            detections.append({
                "label": "speed_sign",
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "speed": speed,
            })

    return detections


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--scene_dir", type=str, default=None)
    args = parser.parse_args()

    reader = init_ocr_reader()

    if args.image:
        frame = cv2.imread(args.image)
        signs = detect_speed_signs(frame, reader)
        for s in signs:
            print(f"  Speed={s['speed']} conf={s['confidence']:.2f} bbox={s['bbox']}")
        if not signs:
            print("No speed signs detected")
    elif args.scene_dir:
        frames_dir = os.path.join(args.scene_dir, "frames")
        for fname in sorted(os.listdir(frames_dir)):
            frame = cv2.imread(os.path.join(frames_dir, fname))
            if frame is None: continue
            signs = detect_speed_signs(frame, reader)
            for s in signs:
                print(f"  {fname}: speed={s['speed']} conf={s['confidence']:.2f}")
