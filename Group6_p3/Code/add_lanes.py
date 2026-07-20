import cv2
import json
import os
import sys
import argparse

# Setup paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "CLRerNet"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "CLRerNet", "libs", "models", "layers", "nms", "src"))

from lane_detect import LaneDetector

DATA_DIR   = os.path.join(PROJECT_DIR, "P3Data")
SEQ_DIR    = os.path.join(DATA_DIR, "Sequences")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

ALL_SCENES = [f"scene{i}" for i in range(1, 14)]


def find_video(undist_dir, camera):
    """Find undistorted video for a camera."""
    for f in os.listdir(undist_dir):
        if camera in f and f.endswith(".mp4"):
            return os.path.join(undist_dir, f)
    return None


def add_lanes_to_scene(scene_name: str, detector: LaneDetector):
    """
    Load existing detections.json, add lane detections, save back.
    """
    det_path = os.path.join(OUTPUT_DIR, scene_name, "detections.json")
    if not os.path.exists(det_path):
        print(f"[{scene_name}] No detections.json found — run detect.py first")
        return

    with open(det_path) as f:
        all_frames = json.load(f)

    # check if lanes already filled
    has_lanes = any(len(fr.get("lanes", [])) > 0 for fr in all_frames)
    if False and has_lanes:  # force rerun
        print(f"[{scene_name}] Lanes already present — skipping")
        return

    # open front camera video
    undist_dir = os.path.join(SEQ_DIR, scene_name, "Undist")
    if not os.path.exists(undist_dir):
        print(f"[{scene_name}] No Undist directory found")
        return

    vid_path = find_video(undist_dir, "front")
    if not vid_path:
        print(f"[{scene_name}] No front video found")
        return

    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{scene_name}] Processing {len(all_frames)} frames from {vid_path}")

    # build frame_idx -> frame lookup
    frame_indices = {fr["frame_idx"]: i for i, fr in enumerate(all_frames)}

    current_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_idx in frame_indices:
            lanes = detector.detect(frame)
            all_frames[frame_indices[current_idx]]["lanes"] = lanes
            processed += 1
            print(f"  Frame {current_idx}/{total}: {len(lanes)} lanes detected")

        current_idx += 1

    cap.release()

    # save back
    with open(det_path, "w") as f:
        json.dump(all_frames, f, indent=2)

    print(f"[{scene_name}] Done — {processed} frames updated, saved to {det_path}")


def main():
    parser = argparse.ArgumentParser(description="Add CLRerNet lanes to detections.json")
    parser.add_argument("--scene", type=str, default="scene1",
                        help="Scene name (e.g. scene1) or 'all' for all 13 scenes")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print("Loading CLRerNet lane detector...")
    detector = LaneDetector(device=args.device)
    print("Detector ready.\n")

    if args.scene == "all":
        scenes = ALL_SCENES
    else:
        scenes = [args.scene]

    for scene in scenes:
        add_lanes_to_scene(scene, detector)

    print("\nAll done!")


if __name__ == "__main__":
    main()
