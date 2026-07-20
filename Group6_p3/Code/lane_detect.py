import cv2
import numpy as np
import os
import sys
import torch
from mmdet.apis import init_detector

# CLRerNet paths
CLRERNET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLRerNet")
NMS_SRC      = os.path.join(CLRERNET_DIR, "libs", "models", "layers", "nms", "src")

sys.path.insert(0, CLRERNET_DIR)
sys.path.insert(0, NMS_SRC)

WEIGHTS = os.path.join(CLRERNET_DIR, "work_dirs", "clrernet",
                       "clrernet_culane_dla34_ema.pth")
CONFIG  = os.path.join(CLRERNET_DIR, "configs", "clrernet", "culane",
                       "clrernet_culane_dla34_ema.py")

# CULane native resolution
CULANE_W, CULANE_H = 1640, 590

# Your Tesla camera native resolution
TESLA_W,  TESLA_H  = 1280, 960


class LaneDetector:
    """
    Wrapper around CLRerNet for per-frame lane detection.
    Instantiate once, call detect() per frame — no model reload overhead.
    """

    def __init__(self, weights=WEIGHTS, config=CONFIG, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        # CLRerNet needs this file to exist for dataset metainfo
        list_path = os.path.join(CLRERNET_DIR, "dataset", "culane", "list")
        os.makedirs(list_path, exist_ok=True)
        test_txt = os.path.join(list_path, "test.txt")
        if not os.path.exists(test_txt):
            open(test_txt, "w").close()

        print(f"[LaneDetector] Loading CLRerNet on {device} ...")
        self.model = init_detector(config, weights, device=device)
        print("[LaneDetector] Model ready.")

    # Public API

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Run CLRerNet on one BGR frame (as returned by cv2.VideoCapture).
        Returns list of lane dicts matching detections.json schema.
        """
        # crop + pad to CULane dimensions
        culane_img, crop_y, pad_x = self._to_culane(frame_bgr)

        # CLRerNet needs a file path
        tmp_path = "/tmp/_clrernet_input.jpg"
        cv2.imwrite(tmp_path, culane_img)

        try:
            from libs.api.inference import inference_one_image
            _, preds = inference_one_image(self.model, tmp_path)
        except Exception as e:
            print(f"[LaneDetector] Inference error: {e}")
            return []

        # convert back to original frame coords and classify
        lanes_orig = self._to_original_coords(preds, crop_y, pad_x, frame_bgr.shape)
        self._current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lanes = []
        for points in lanes_orig:
            if len(points) < 2:
                continue
            color = self._classify_color(frame_bgr, points)
            ltype = self._classify_type(points)
            lanes.append({
                "points": [[int(x), int(y)] for x, y in points],
                "color":  color,
                "type":   ltype,
            })

        return lanes

    # Pre/post processing

    def _to_culane(self, frame: np.ndarray):
        """
        Convert Tesla frame (1280x960) to CULane dims (1640x590).
        Crop vertical centre, pad horizontally to avoid aspect ratio distortion.
        Returns (culane_img, crop_y, pad_x).
        """
        h, w = frame.shape[:2]

        # Step 1: Vertical crop to 590px from centre
        crop_h = CULANE_H                          # 590
        crop_y = max(0, (h - crop_h) // 2)        # ~185px from top for 960h
        cropped = frame[crop_y: crop_y + crop_h, :]   # (590, 1280, 3)

        # Step 2: Horizontal pad to 1640px
        pad_total = CULANE_W - w                   # 1640 - 1280 = 360
        pad_x     = pad_total // 2                 # 180px each side
        culane_img = cv2.copyMakeBorder(
            cropped, 0, 0, pad_x, pad_total - pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return culane_img, crop_y, pad_x

    def _to_original_coords(self, preds, crop_y: int, pad_x: int,
                             orig_shape: tuple) -> list:
        """
        Invert the crop+pad transform to get points in original frame coords.

        preds from CLRerNet: list of lanes, each lane is list of (x, y)
        tuples in CULane (1640x590) space.
        """
        lanes = []
        if preds is None:
            return lanes

        for lane in preds:
            points = []
            for x, y in lane:
                # remove horizontal padding
                x_orig = x - pad_x
                # remove vertical crop offset
                y_orig = y + crop_y
                # clip to original frame bounds
                h, w = orig_shape[:2]
                if 0 <= x_orig < w and 0 <= y_orig < h:
                    points.append((x_orig, y_orig))
            if len(points) >= 2:
                points.sort(key=lambda p: p[1])
                lanes.append(points)
        return lanes

    # Color classification (HSV sampling along lane)

    def _classify_color(self, frame: np.ndarray, points: list,
                         sample_n: int = 20) -> str:
        """
        Sample neighborhood (+-6px) around lane points in HSV.
        Pick brightest pixel in patch - that is the paint, not road surface.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_img, w_img = frame.shape[:2]
        step = max(1, len(points) // sample_n)
        sampled = points[::step]
        yellow_votes = 0
        white_votes = 0
        RADIUS = 6
        for px, py in sampled:
            x, y = int(px), int(py)
            best_h, best_s, best_v = 0, 0, 0
            for dy in range(-RADIUS, RADIUS + 1, 2):
                for dx in range(-RADIUS, RADIUS + 1, 2):
                    yi, xi = y + dy, x + dx
                    if not (0 <= yi < h_img and 0 <= xi < w_img):
                        continue
                    hv, sv, vv = hsv[yi, xi]
                    if vv > best_v:
                        best_h, best_s, best_v = int(hv), int(sv), int(vv)
            if best_v < 80:
                continue
            if 15 <= best_h <= 40 and best_s > 50 and best_v > 100:
                yellow_votes += 1
            elif best_s < 40 and best_v > 180:
                white_votes += 1
        if yellow_votes > white_votes and yellow_votes >= 2:
            return "yellow"
        return "white"


    def _classify_type(self, points: list,
                        min_transitions: int = 4) -> str:
        """
        CLRerNet interpolates through dashed gaps. Check brightness along
        polyline: paint=bright, road=dark. Count transitions.
        """
        if len(points) < 10 or not hasattr(self, '_current_gray'):
            return "solid"
        gray = self._current_gray
        h_img, w_img = gray.shape[:2]
        step = max(1, len(points) // 40)
        sampled = points[::step]
        bright = []
        for x, y in sampled:
            xi, yi = int(x), int(y)
            if 0 <= yi < h_img and 0 <= xi < w_img:
                patch = gray[max(0,yi-3):min(h_img,yi+4), max(0,xi-3):min(w_img,xi+4)]
                bright.append(float(patch.max()) if patch.size > 0 else 0)
        if len(bright) < 10:
            return "solid"
        med = sorted(bright)[len(bright) // 2]
        threshold = max(med + 20, 140)
        is_paint = [b > threshold for b in bright]
        transitions = sum(1 for i in range(1, len(is_paint)) if is_paint[i] != is_paint[i-1])
        return "dashed" if transitions >= min_transitions else "solid"


_detector = None   


def detect_lanes_clrernet(frame_bgr: np.ndarray, device=None) -> list:
    """
    Drop-in replacement for the empty "lanes": [] in detect.py.

    """
    if frame_bgr is None:
        return []

    global _detector
    if _detector is None:
        _detector = LaneDetector(device=device)

    return _detector.detect(frame_bgr)



if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description="Test lane_detect.py on a single image")
    parser.add_argument("--image",  type=str, required=True,
                        help="Path to input image (BGR, any resolution)")
    parser.add_argument("--out",    type=str, default="lane_output.jpg",
                        help="Output image with lanes drawn")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        print(f"ERROR: Cannot read {args.image}")
        sys.exit(1)

    detector = LaneDetector(device=args.device)
    lanes    = detector.detect(frame)

    print(f"\nDetected {len(lanes)} lanes:")
    print(json.dumps(lanes, indent=2))

    # draw on frame for visual verification
    for lane in lanes:
        pts       = np.array(lane["points"], dtype=np.int32)
        color_bgr = (0, 255, 255) if lane["color"] == "yellow" else (255, 255, 255)
        thickness = 3 if lane["type"] == "solid" else 2

        if lane["type"] == "dashed":
            for i in range(0, len(pts) - 1, 2):
                cv2.line(frame, tuple(pts[i]),
                         tuple(pts[min(i + 1, len(pts) - 1)]),
                         color_bgr, thickness)
        else:
            cv2.polylines(frame, [pts], False, color_bgr, thickness)

        # label mid-lane
        if len(pts) > 0:
            mid = pts[len(pts) // 2]
            cv2.putText(frame, f"{lane['color']} {lane['type']}",
                        tuple(mid), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color_bgr, 2)

    cv2.imwrite(args.out, frame)
    print(f"\nSaved annotated frame to: {args.out}")
