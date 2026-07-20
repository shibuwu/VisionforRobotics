"""
Main detection pipeline (Phase 1 + 2).
YOLO11 detection with ByteTrack tracking, optical flow heading,
traffic light color/arrow, speed sign OCR, and road arrow detection.
"""

import cv2
import numpy as np
import os
import json
import torch
from ultralytics import YOLO
from classify_arrow import classify_arrow
from detect_speed_sign import detect_speed_signs, init_ocr_reader
from detect_road_arrow import detect_road_arrows

def select_device(device_str):
    if device_str == 'cpu':
        return torch.device('cpu')
    return torch.device(f'cuda:{device_str}' if torch.cuda.is_available() else 'cpu')

# paths
DATA_DIR = "P3Data"
SEQ_DIR = os.path.join(DATA_DIR, "Sequences")
OUTPUT_DIR = "output"



# YOLO classes we care about (COCO)
CLASSES_OF_INTEREST = {
    0: "pedestrian",
    1: "vehicle",      # bicycle
    2: "vehicle",      # car
    3: "vehicle",      # motorcycle
    5: "vehicle",      # bus
    7: "vehicle",      # truck
    9: "traffic_light",
    11: "stop_sign",
}

# COCO class → vehicle subtype (set at detection time)
COCO_VEHICLE_TYPE = {
    1: "bicycle",
    3: "motorcycle",
    5: "truck",
    7: "truck",
}


def classify_tl_color(frame, bbox):
    """Classify traffic light color using spatial brightness.
    Splits bbox into top/middle/bottom thirds — the lit bulb is the brightest.
    Red=top, yellow=middle, green=bottom."""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 6:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h3 = crop.shape[0] // 3
    thirds = [hsv[:h3, :], hsv[h3:2*h3, :], hsv[2*h3:, :]]

    brightness = []
    for t in thirds:
        sat_mask = t[:, :, 1] > 50
        if sat_mask.sum() > 0:
            brightness.append(float(t[:, :, 2][sat_mask].mean()))
        else:
            brightness.append(0)

    brightest = int(np.argmax(brightness))
    if brightness[brightest] > 25:
        return ["red", "yellow", "green"][brightest]
    return "unknown"


def get_depth_for_bbox(depth_map, bbox, frame_shape):
    """Get median depth value within a bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    dh, dw = depth_map.shape
    fh, fw = frame_shape[:2]

    # scale bbox to depth map size
    sx1 = int(x1 * dw / fw)
    sy1 = int(y1 * dh / fh)
    sx2 = int(x2 * dw / fw)
    sy2 = int(y2 * dh / fh)

    sx1, sy1 = max(0, sx1), max(0, sy1)
    sx2, sy2 = min(dw, sx2), min(dh, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        return 0.0

    region = depth_map[sy1:sy2, sx1:sx2]
    return float(np.median(region))


# confidence thresholds per class (higher = fewer false positives)
CONF_THRESHOLDS = {
    "vehicle": 0.45,
    "pedestrian": 0.50,
    "traffic_light": 0.30,
    "stop_sign": 0.55,
}


def estimate_of_direction(flow, bbox, frame_shape):
    """Use optical flow to determine if vehicle is same-direction or oncoming.
    Returns 'same', 'oncoming', or 'unknown'."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_shape[:2]
    roi = flow[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        return "unknown"

    veh_flow = np.median(roi, axis=(0, 1))
    # vector from FOE (center of image) to bbox center
    foe = np.array([w / 2, h / 2])
    bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    to_bbox = bbox_center - foe
    norm = np.linalg.norm(to_bbox)
    if norm < 1:
        return "unknown"

    # radial component: positive = expanding (same dir), negative = contracting (oncoming)
    radial = np.dot(veh_flow, to_bbox / norm)
    bbox_h = y2 - y1

    # only trust OF for close vehicles (large bbox)
    if bbox_h < 60:
        return "unknown"

    if radial > 3.0:
        return "same"
    elif radial < -3.0:
        return "oncoming"
    return "unknown"


def bbox_iou(box1, box2):
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def filter_stop_sign(det, frame):
    """Filter out likely false positive stop signs."""
    x1, y1, x2, y2 = det["bbox"]
    h, w = frame.shape[:2]

    # stop signs should be in upper 70% of frame
    if y1 > h * 0.7:
        return False

    # roughly square aspect ratio
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    ratio = bbox_w / max(bbox_h, 1)
    if ratio < 0.5 or ratio > 2.0:
        return False

    return True


def process_frame(yolo_model, frame, frame_idx, depth_map, device,
                  use_tracking=False):
    """Run all detections on a single frame."""
    # YOLO object detection (with or without tracking)
    if use_tracking:
        results = yolo_model.track(frame, persist=True, verbose=False, imgsz=1280)[0]
    else:
        results = yolo_model(frame, verbose=False, imgsz=1280)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in CLASSES_OF_INTEREST:
            continue

        label = CLASSES_OF_INTEREST[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])

        # confidence filter
        if conf < CONF_THRESHOLDS.get(label, 0.3):
            continue

        det = {
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
        }

        # tag vehicle subtype from COCO class
        if cls_id in COCO_VEHICLE_TYPE:
            det["vehicle_type"] = COCO_VEHICLE_TYPE[cls_id]

        # traffic light: classify color using spatial brightness
        if label == "traffic_light":
            bbox_h = y2 - y1
            bbox_w = x2 - x1
            if y1 > frame.shape[0] * 0.6:
                continue
            if bbox_h < bbox_w * 0.8:
                continue
            det["color"] = classify_tl_color(frame, [x1, y1, x2, y2])
            det["arrow"] = classify_arrow(frame, [x1, y1, x2, y2], det["color"])

        # track ID if available
        if use_tracking and box.id is not None:
            det["track_id"] = int(box.id[0])

        # depth
        if depth_map is not None:
            det["depth"] = get_depth_for_bbox(depth_map, [x1, y1, x2, y2], frame.shape)

        # stop sign filtering
        if label == "stop_sign":
            if not filter_stop_sign(det, frame):
                continue

        detections.append(det)

    # cyclist/rider detection: if a pedestrian bbox overlaps significantly with a
    # bicycle or motorcycle bbox, mark the pedestrian as a rider and skip rendering them
    bikes = [d for d in detections if d.get("vehicle_type") in ("bicycle", "motorcycle")]
    peds = [d for d in detections if d["label"] == "pedestrian"]
    for ped in peds:
        for bike in bikes:
            bx1 = max(ped["bbox"][0], bike["bbox"][0])
            by1 = max(ped["bbox"][1], bike["bbox"][1])
            bx2 = min(ped["bbox"][2], bike["bbox"][2])
            by2 = min(ped["bbox"][3], bike["bbox"][3])
            inter = max(0, bx2 - bx1) * max(0, by2 - by1)
            bike_area = (bike["bbox"][2] - bike["bbox"][0]) * (bike["bbox"][3] - bike["bbox"][1])
            if bike_area > 0 and inter / bike_area > 0.3:
                ped["is_cyclist"] = True
                break

    return {
        "frame_idx": frame_idx,
        "detections": detections,
        "lanes": [],
    }


CAM_VIDEO_KEYS = {
    "front": "front",
    "back": "back",
    "left": "left_repeater",
    "right": "right_repeater",
}


def find_video(undist_dir, cam_name):
    key = CAM_VIDEO_KEYS[cam_name]
    for f in os.listdir(undist_dir):
        if key in f and f.endswith(".mp4"):
            return os.path.join(undist_dir, f)
    return None


def smooth_headings(all_results, iou_threshold=0.3):
    """Smooth vehicle headings: link detections into tracks by IoU,
    then apply median heading per track to eliminate flickering."""

    # Pass 1: link detections across frames into tracks
    track_counter = 0
    prev_dets = []
    for frame_data in all_results:
        curr_dets = [d for d in frame_data["detections"] if d["label"] == "vehicle"]
        used = set()
        for curr in curr_dets:
            best_iou, best_idx = 0, -1
            for pi, prev in enumerate(prev_dets):
                if pi in used or prev.get("camera") != curr.get("camera"):
                    continue
                iou = bbox_iou(curr["bbox"], prev["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, pi
            if best_idx >= 0 and best_iou > iou_threshold:
                curr["_tid"] = prev_dets[best_idx]["_tid"]
                used.add(best_idx)
            else:
                curr["_tid"] = track_counter
                track_counter += 1
        prev_dets = curr_dets

    # Pass 2: median heading per track
    tracks = {}
    for frame_data in all_results:
        for d in frame_data["detections"]:
            if "_tid" not in d:
                continue
            tid = d["_tid"]
            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append(d)

    smoothed = 0
    for tid, dets in tracks.items():
        headings = [d.get("heading", 0) for d in dets]
        # majority vote: snap each heading to 0 (away) or pi (toward),
        # then pick whichever direction has more votes
        away_count = sum(1 for h in headings if abs(h) < np.pi / 2)
        toward_count = len(headings) - away_count
        consensus_h = 0.0 if away_count >= toward_count else np.pi
        for d in dets:
            if abs(d.get("heading", 0) - consensus_h) > 0.3:
                smoothed += 1
            d["heading"] = float(consensus_h)
            d["heading_source"] = "smoothed"
            del d["_tid"]

    print(f"  Smoothed {smoothed} flips across {len(tracks)} tracks")


def smooth_arrows(all_results, iou_threshold=0.3):
    """Majority-vote arrow direction per traffic light track."""
    track_counter = 0
    prev_dets = []
    for frame_data in all_results:
        curr_dets = [d for d in frame_data["detections"]
                     if d["label"] == "traffic_light"]
        used = set()
        for curr in curr_dets:
            best_iou, best_idx = 0, -1
            for pi, prev in enumerate(prev_dets):
                if pi in used or prev.get("camera") != curr.get("camera"):
                    continue
                iou = bbox_iou(curr["bbox"], prev["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, pi
            if best_idx >= 0 and best_iou > iou_threshold:
                curr["_atid"] = prev_dets[best_idx]["_atid"]
                used.add(best_idx)
            else:
                curr["_atid"] = track_counter
                track_counter += 1
        prev_dets = curr_dets

    tracks = {}
    for frame_data in all_results:
        for d in frame_data["detections"]:
            if "_atid" not in d:
                continue
            tracks.setdefault(d["_atid"], []).append(d)

    smoothed = 0
    for tid, dets in tracks.items():
        counts = {}
        for d in dets:
            a = d.get("arrow", "none")
            counts[a] = counts.get(a, 0) + 1
        consensus = max(counts, key=counts.get)
        for d in dets:
            if d.get("arrow", "none") != consensus:
                smoothed += 1
            d["arrow"] = consensus
            del d["_atid"]

    print(f"  Smoothed {smoothed} arrow flips across {len(tracks)} TL tracks")


def smooth_speed_signs(all_results, iou_threshold=0.3):
    """Majority-vote speed value per speed sign track."""
    track_counter = 0
    prev_dets = []
    for frame_data in all_results:
        curr_dets = [d for d in frame_data["detections"]
                     if d["label"] == "speed_sign"]
        used = set()
        for curr in curr_dets:
            best_iou, best_idx = 0, -1
            for pi, prev in enumerate(prev_dets):
                if pi in used or prev.get("camera") != curr.get("camera"):
                    continue
                iou = bbox_iou(curr["bbox"], prev["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, pi
            if best_idx >= 0 and best_iou > iou_threshold:
                curr["_stid"] = prev_dets[best_idx]["_stid"]
                used.add(best_idx)
            else:
                curr["_stid"] = track_counter
                track_counter += 1
        prev_dets = curr_dets

    tracks = {}
    for frame_data in all_results:
        for d in frame_data["detections"]:
            if "_stid" not in d:
                continue
            tracks.setdefault(d["_stid"], []).append(d)

    smoothed = 0
    for tid, dets in tracks.items():
        counts = {}
        for d in dets:
            s = d.get("speed", 0)
            counts[s] = counts.get(s, 0) + 1
        consensus = max(counts, key=counts.get)
        for d in dets:
            if d.get("speed", 0) != consensus:
                smoothed += 1
            d["speed"] = consensus
            del d["_stid"]

    if tracks:
        print(f"  Smoothed {smoothed} speed sign flips across {len(tracks)} tracks")


def smooth_road_arrows(all_results):
    """Majority-vote road arrow direction per track (matched by world-space proximity)."""
    track_counter = 0
    prev_dets = []
    for frame_data in all_results:
        curr_dets = [d for d in frame_data["detections"]
                     if d["label"] == "road_arrow"]
        used = set()
        for curr in curr_dets:
            best_dist, best_idx = 4.0, -1
            for pi, prev in enumerate(prev_dets):
                if pi in used:
                    continue
                dist = np.linalg.norm(np.array(curr["world_pos"][:2]) -
                                      np.array(prev["world_pos"][:2]))
                if dist < best_dist:
                    best_dist, best_idx = dist, pi
            if best_idx >= 0:
                curr["_raid"] = prev_dets[best_idx]["_raid"]
                used.add(best_idx)
            else:
                curr["_raid"] = track_counter
                track_counter += 1
        prev_dets = curr_dets

    tracks = {}
    for frame_data in all_results:
        for d in frame_data["detections"]:
            if "_raid" not in d:
                continue
            tracks.setdefault(d["_raid"], []).append(d)

    smoothed = 0
    for tid, dets in tracks.items():
        counts = {}
        for d in dets:
            dr = d.get("direction", "straight")
            counts[dr] = counts.get(dr, 0) + 1
        consensus = max(counts, key=counts.get)
        for d in dets:
            if d.get("direction", "straight") != consensus:
                smoothed += 1
            d["direction"] = consensus
            del d["_raid"]

    if tracks:
        print(f"  Smoothed {smoothed} road arrow flips across {len(tracks)} tracks")


def process_scene(scene_name, yolo_model, device, sample_rate=36,
                  only_frames=None):
    """Process a scene using front camera with ByteTrack tracking."""
    scene_dir = os.path.join(SEQ_DIR, scene_name)
    undist_dir = os.path.join(scene_dir, "Undist")

    scene_out = os.path.join(OUTPUT_DIR, scene_name)
    os.makedirs(os.path.join(scene_out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(scene_out, "annotated"), exist_ok=True)

    vid = find_video(undist_dir, "front")
    if not vid:
        print(f"No front video for {scene_name}, skipping")
        return

    cap = cv2.VideoCapture(vid)
    print(f"Processing {scene_name}: {vid}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    depth_dir = os.path.join(OUTPUT_DIR, scene_name, "depth")
    has_depth = os.path.exists(depth_dir)

    calib_raw = json.load(open(os.path.join(DATA_DIR, "Calib", "calibration_results.json")))

    ocr_reader = init_ocr_reader()

    K_front_arr = np.array(calib_raw["front"]["K"])
    ext_path = os.path.join(DATA_DIR, "Calib", "extrinsics.json")
    if os.path.exists(ext_path):
        ext_front_data = json.load(open(ext_path))["front"]
    else:
        ext_front_data = None

    tracker = YOLO(yolo_model.ckpt_path)

    all_results = []
    frame_idx = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        should_process = (frame_idx % sample_rate == 0) if only_frames is None else (frame_idx in only_frames)
        if should_process:
            depth_map = None
            if has_depth:
                depth_path = os.path.join(depth_dir, f"depth_{frame_idx:05d}.npy")
                if os.path.exists(depth_path):
                    depth_map = np.load(depth_path)

            # optical flow for heading estimation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = None
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray = gray

            result = process_frame(tracker, frame, frame_idx, depth_map,
                                   device, use_tracking=True)

            for det in result["detections"]:
                det["camera"] = "front"

            # heading from optical flow (FCOS3D overwrites later in pipeline)
            for det in result["detections"]:
                if det["label"] == "vehicle" and "heading" not in det:
                    yaw = 0.0  # default: same direction as ego
                    if flow is not None:
                        of_dir = estimate_of_direction(flow, det["bbox"], frame.shape)
                        if of_dir == "oncoming":
                            yaw = np.pi
                    det["heading"] = float(yaw)
                    det["heading_source"] = "of"

            all_detections = list(result["detections"])

            # speed sign detection (OCR)
            if ocr_reader is not None:
                speed_dets = detect_speed_signs(frame, ocr_reader)
                for sd in speed_dets:
                    sd["camera"] = "front"
                    if depth_map is not None:
                        sd["depth"] = get_depth_for_bbox(depth_map, sd["bbox"], frame.shape)
                all_detections.extend(speed_dets)

            # road arrow detection (BEV projection)
            if ext_front_data is not None:
                ra_dets = detect_road_arrows(
                    frame, K_front_arr,
                    ext_front_data["R"], ext_front_data["t"])
                for rd in ra_dets:
                    rd["camera"] = "front"
                all_detections.extend(ra_dets)

            # within-frame IoU dedup
            keep = [True] * len(all_detections)
            for i in range(len(all_detections)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(all_detections)):
                    if not keep[j]:
                        continue
                    if all_detections[i]["label"] != all_detections[j]["label"]:
                        continue
                    iou = bbox_iou(all_detections[i]["bbox"], all_detections[j]["bbox"])
                    if iou > 0.3:
                        if all_detections[i]["confidence"] >= all_detections[j]["confidence"]:
                            keep[j] = False
                        else:
                            keep[i] = False
                            break
            all_detections = [d for d, k in zip(all_detections, keep) if k]

            frame_result = {
                "frame_idx": frame_idx,
                "detections": all_detections,
                "lanes": [],
            }
            all_results.append(frame_result)

            cv2.imwrite(
                os.path.join(scene_out, "frames", f"frame_{frame_idx:05d}.jpg"), frame)
            annotated = draw_detections(frame.copy(), frame_result)
            cv2.imwrite(
                os.path.join(scene_out, "annotated", f"frame_{frame_idx:05d}.jpg"), annotated)

            print(f"  Frame {frame_idx}/{total}: {len(all_detections)} objects")

        frame_idx += 1

    cap.release()

    # post-process: smooth across frames using bbox tracking
    smooth_headings(all_results)
    smooth_arrows(all_results)
    smooth_speed_signs(all_results)
    smooth_road_arrows(all_results)

    with open(os.path.join(scene_out, "detections.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done {scene_name}: {len(all_results)} frames processed")
    return all_results


def draw_detections(frame, result):
    """Draw bounding boxes and lanes on frame for debugging."""
    colors = {
        "vehicle": (0, 255, 0),
        "pedestrian": (255, 0, 0),
        "traffic_light": (0, 255, 255),
        "stop_sign": (0, 0, 255),
    }

    for det in result["detections"]:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = colors.get(det["label"], (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = det["label"]
        if det["label"] == "traffic_light" and "color" in det:
            arrow_str = det.get("arrow", "none")
            if arrow_str != "none":
                label += f" ({det['color']}, {arrow_str})"
            else:
                label += f" ({det['color']})"
        elif det["label"] == "speed_sign" and "speed" in det:
            label += f" ({det['speed']} mph)"
        if "depth" in det:
            label += f" d={det['depth']:.1f}"
        cv2.putText(frame, f"{label} {det['confidence']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for lane in result["lanes"]:
        pts = np.array(lane["points"], dtype=np.int32)
        lane_color = (0, 255, 255) if lane["color"] == "yellow" else (255, 255, 255)
        if lane["type"] == "dashed":
            # draw dashed
            for i in range(0, len(pts) - 1, 2):
                cv2.line(frame, tuple(pts[i]), tuple(pts[min(i+1, len(pts)-1)]), lane_color, 2)
        else:
            cv2.polylines(frame, [pts], False, lane_color, 2)

    return frame


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene1")
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--frames", type=str, default=None,
                        help="Comma-separated frame indices to process (e.g. 905,910,920)")
    args = parser.parse_args()

    device = select_device('0' if torch.cuda.is_available() else 'cpu')

    print("Loading YOLO model...")
    yolo_model = YOLO("yolo11m.pt")

    only_frames = None
    if args.frames:
        only_frames = set(int(x) for x in args.frames.split(","))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_scene(args.scene, yolo_model, device, args.sample_rate,
                  only_frames=only_frames)
