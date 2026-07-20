# Vehicle heading estimation — copies FCOS3D yaw into heading field
# with per-track smoothing, gap-filling, and propagation.
# Convention: 0 = away from ego, pi = toward ego.

import os
import json
import argparse

OUTPUT_DIR = "output"

def estimate_headings_for_scene(scene):
    det_path = os.path.join(OUTPUT_DIR, scene, "detections.json")
    if not os.path.exists(det_path):
        print(f"    skip: no detections.json")
        return

    with open(det_path) as f:
        frames_data = json.load(f)

    total, fcos3d_count, default_count = 0, 0, 0

    for e in frames_data:
        for det in e.get("detections", []):
            if det["label"] != "vehicle":
                continue
            total += 1

            if "fcos3d_heading" in det:
                det["heading"] = det["fcos3d_heading"]
                det["heading_source"] = "fcos3d"
                det["heading_confidence"] = min(0.95, det.get("fcos3d_score", 0.5))
                fcos3d_count += 1
            else:
                det["heading"] = 0.0
                det["heading_source"] = "default"
                det["heading_confidence"] = 0.0
                default_count += 1

    # track-based smoothing: group by track_id
    import math
    from collections import Counter
    tracks = {}
    for ei, e in enumerate(frames_data):
        for di, det in enumerate(e.get("detections", [])):
            if det["label"] != "vehicle":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            tracks.setdefault(tid, []).append((ei, di))

    yaw_smoothed = 0
    type_smoothed = 0
    color_smoothed = 0

    for tid, entries in tracks.items():
        if len(entries) < 2:
            continue

        # collect per-track data
        dets = [frames_data[ei]["detections"][di] for ei, di in entries]

        # 1. smooth yaw: moving median window=5
        yaws = [d.get("fcos3d_yaw") for d in dets]
        for i, (ei, di) in enumerate(entries):
            if yaws[i] is None:
                continue
            window = [y for y in yaws[max(0,i-2):i+3] if y is not None]
            if not window:
                continue
            med = sorted(window)[len(window)//2]
            if abs(yaws[i] - med) > 0.3:
                det = frames_data[ei]["detections"][di]
                det["fcos3d_yaw"] = med
                det["heading"] = det["fcos3d_heading"] = med - math.pi / 2
                yaw_smoothed += 1

        # 2. smooth vehicle_type: sliding window=10 majority vote
        types = [d.get("vehicle_type", "sedan") for d in dets]
        for i, (ei, di) in enumerate(entries):
            window = types[max(0, i-5):i+6]
            maj = Counter(window).most_common(1)[0][0]
            det = frames_data[ei]["detections"][di]
            if det.get("vehicle_type") != maj:
                det["vehicle_type"] = maj
                type_smoothed += 1

        # 3. smooth vehicle_color: sliding window=10 majority vote
        colors = [d.get("vehicle_color", "gray") for d in dets]
        for i, (ei, di) in enumerate(entries):
            window = colors[max(0, i-5):i+6]
            maj = Counter(window).most_common(1)[0][0]
            det = frames_data[ei]["detections"][di]
            if det.get("vehicle_color") != maj:
                det["vehicle_color"] = maj
                color_smoothed += 1

    # 4. gap-filling: if a track has frames before+after a gap, interpolate
    fidx_to_ei = {e["frame_idx"]: ei for ei, e in enumerate(frames_data)}
    all_fidxs = sorted(fidx_to_ei.keys())
    step = all_fidxs[1] - all_fidxs[0] if len(all_fidxs) > 1 else 10
    filled = 0

    # rebuild tracks with frame indices
    track_frames = {}
    for ei, e in enumerate(frames_data):
        fidx = e["frame_idx"]
        for di, det in enumerate(e.get("detections", [])):
            if det["label"] != "vehicle" or det.get("heading_source") != "fcos3d":
                continue
            if det.get("confidence", 0) < 0.6:
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            track_frames.setdefault(tid, {})[fidx] = det

    for tid, frame_dets in track_frames.items():
        fidxs = sorted(frame_dets.keys())
        if len(fidxs) < 2:
            continue
        for i in range(len(fidxs) - 1):
            gap_start = fidxs[i]
            gap_end = fidxs[i + 1]
            # fill gaps up to ~10 base steps
            if gap_end - gap_start <= step or gap_end - gap_start > step * 10:
                continue
            det_before = frame_dets[gap_start]
            det_after = frame_dets[gap_end]
            # interpolate for each missing frame
            for missing_fidx in range(gap_start + step, gap_end, step):
                if missing_fidx not in fidx_to_ei:
                    continue
                mei = fidx_to_ei[missing_fidx]
                # lerp bbox
                t = (missing_fidx - gap_start) / (gap_end - gap_start)
                bbox = [det_before["bbox"][j] * (1 - t) + det_after["bbox"][j] * t for j in range(4)]
                interp = {
                    "label": "vehicle",
                    "bbox": bbox,
                    "confidence": min(det_before["confidence"], det_after["confidence"]),
                    "camera": "front",
                    "track_id": tid,
                    "heading_source": "fcos3d",
                    "heading": det_before.get("heading", 0),
                    "fcos3d_yaw": det_before.get("fcos3d_yaw", 0),
                    "fcos3d_heading": det_before.get("fcos3d_heading", 0),
                    "fcos3d_depth": det_before.get("fcos3d_depth", 0) * (1-t) + det_after.get("fcos3d_depth", 0) * t,
                    "fcos3d_pos": [det_before["fcos3d_pos"][j] * (1-t) + det_after["fcos3d_pos"][j] * t for j in range(3)] if det_before.get("fcos3d_pos") and det_after.get("fcos3d_pos") else det_before.get("fcos3d_pos"),
                    "fcos3d_score": min(det_before.get("fcos3d_score", 0), det_after.get("fcos3d_score", 0)),
                    "vehicle_type": det_before.get("vehicle_type", "sedan"),
                    "vehicle_color": det_before.get("vehicle_color", "gray"),
                    "fcos3d_class": det_before.get("fcos3d_class"),
                }
                frames_data[mei]["detections"].append(interp)
                filled += 1

    track_all = {}  # tid -> list of (ei, di, det, fidx)
    for ei, e in enumerate(frames_data):
        fidx = e["frame_idx"]
        for di, det in enumerate(e.get("detections", [])):
            if det["label"] != "vehicle":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            track_all.setdefault(tid, []).append((ei, di, det, fidx))

    propagated = 0
    for tid, entries in track_all.items():
        # find fcos3d donor frames in this track
        donors = [(fi, det) for (ei, di, det, fi) in entries
                  if det.get("heading_source") == "fcos3d" and "fcos3d_yaw" in det]
        if not donors:
            continue
        donor_fidxs = [d[0] for d in donors]
        for ei, di, det, fidx in entries:
            if det.get("heading_source") == "fcos3d":
                continue
            # nearest donor by frame distance
            nearest = min(donors, key=lambda d: abs(d[0] - fidx))[1]
            det["fcos3d_yaw"] = nearest["fcos3d_yaw"]
            det["fcos3d_heading"] = nearest.get("fcos3d_heading", 0)
            det["fcos3d_dims"] = nearest.get("fcos3d_dims")
            det["heading"] = nearest.get("heading", nearest.get("fcos3d_heading", 0))
            det["heading_source"] = "fcos3d_track"
            det["heading_confidence"] = nearest.get("heading_confidence", 0.5) * 0.8
            propagated += 1

    with open(det_path, "w") as f:
        json.dump(frames_data, f)

    print(f"    {total} vehicles: fcos3d={fcos3d_count} default={default_count}")
    print(f"    smoothed: {yaw_smoothed} yaw, {type_smoothed} type, {color_smoothed} color flips")
    print(f"    filled: {filled} gap frames")
    print(f"    propagated: {propagated} fcos3d_track entries")


def main():
    parser = argparse.ArgumentParser(description="Vehicle heading from FCOS3D")
    parser.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args()

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([
            d for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))
            and os.path.exists(os.path.join(OUTPUT_DIR, d, "detections.json"))
        ])

    print(f"Processing {len(scenes)} scenes...")
    for scene in scenes:
        print(f"  -> {scene}")
        estimate_headings_for_scene(scene)
    print("Done.")


if __name__ == "__main__":
    main()
