import argparse
import json
import os
import numpy as np
from collections import defaultdict


# v4 delta-hold parameters (used by v5)
V4_ONSET_JUMP = 0.15
V4_ONSET_MIN_SCORE = 0.15
V4_HOLD_FRAMES = 30
V4_DROP_JUMP = 0.10
V4_MAX_FRAME_GAP = 3

# v6 bulb brightness baseline parameters
V6_REL_FACTOR = 1.5
V6_ABS_DELTA = 0.15
V6_MIN_SCORE = 0.20
V6_MIN_SAMPLES = 4

# motion filter parameters
MOTION_WINDOW = 15
MOTION_MIN_HEIGHT_GROWTH = 0.003


def smooth_bools(flags, window=3, min_votes=2):
    out = []
    for i in range(len(flags)):
        lo = max(0, i - window // 2)
        hi = min(len(flags), i + window // 2 + 1)
        out.append(sum(flags[lo:hi]) >= min_votes)
    return out


def strategy_delta_hold(items):
    """v4: delta onset + temporal hold on strip-red score."""
    n = len(items)
    if n < 2:
        return [False] * n
    flags = [False] * n
    state_on = False
    hold_until_fidx = -1

    for i in range(n):
        fidx = items[i][0]
        b = items[i][1]

        is_onset = False
        is_offset = False
        if i > 0:
            pf = items[i - 1][0]
            pb = items[i - 1][1]
            gap = fidx - pf
            if gap <= V4_MAX_FRAME_GAP:
                jump = b - pb
                if jump >= V4_ONSET_JUMP and b >= V4_ONSET_MIN_SCORE:
                    is_onset = True
                elif (pb - b) >= V4_DROP_JUMP and pb >= V4_ONSET_MIN_SCORE:
                    is_offset = True

        if is_onset:
            state_on = True
            hold_until_fidx = fidx + V4_HOLD_FRAMES
        elif state_on and is_offset:
            state_on = False
        elif state_on and fidx >= hold_until_fidx:
            state_on = False

        flags[i] = state_on
    return flags


def strategy_bulb_baseline(items):
    """v6: per-track baseline on bulb brightness (V>=200 fraction).
    A pressed brake makes the bulb brighter than its own baseline."""
    n = len(items)
    if n < V6_MIN_SAMPLES:
        return [False] * n
    bulbs = np.array([item[4] if len(item) >= 5 else 0.0 for item in items],
                     dtype=np.float32)
    baseline = float(np.median(bulbs))
    threshold = max(baseline * V6_REL_FACTOR,
                    baseline + V6_ABS_DELTA,
                    V6_MIN_SCORE)
    return [bool(b >= threshold) for b in bulbs]


def build_track_motion(scene):
    """Per-track bbox history from detections.json."""
    det_path = os.path.join("output", scene, "detections.json")
    if not os.path.exists(det_path):
        return {}
    with open(det_path) as f:
        all_frames = json.load(f)

    by_track = defaultdict(list)
    for entry in all_frames:
        fidx = entry["frame_idx"]
        for det in entry.get("detections", []):
            if det.get("label") != "vehicle":
                continue
            if det.get("camera", "front") != "front":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            h = y2 - y1
            by_track[int(tid)].append((fidx, cx, cy, h))
    for tid in by_track:
        by_track[tid].sort()
    return dict(by_track)


def compute_closing_frames(track_history):
    """Return frame indices where bbox height is growing (car approaching ego)."""
    closing = set()
    n = len(track_history)
    for i in range(n):
        fidx_i, _, _, h_i = track_history[i]
        if h_i < 10:
            continue
        past = None
        for j in range(i - 1, -1, -1):
            fj, _, _, hj = track_history[j]
            if fidx_i - fj > MOTION_WINDOW:
                break
            past = (fj, hj)
            if fidx_i - fj >= 5:
                break
        if past is None:
            closing.add(fidx_i)
            continue
        fj, hj = past
        dt = max(fidx_i - fj, 1)
        rel_growth = (h_i - hj) / hj / dt
        if rel_growth >= MOTION_MIN_HEIGHT_GROWTH:
            closing.add(fidx_i)
    return closing


def apply_motion_filter(flags_dict, scene):
    """Keep brake flags only where the track's bbox is closing on ego."""
    motion = build_track_motion(scene)
    closing_per_track = {tid: compute_closing_frames(hist)
                         for tid, hist in motion.items()}

    kept = {}
    dropped = 0
    for (fidx, tid), conf in flags_dict.items():
        closing_set = closing_per_track.get(int(tid), set())
        if fidx not in closing_set:
            dropped += 1
            continue
        kept[(fidx, tid)] = conf
    print(f"  motion filter: dropped {dropped} flags")
    return kept


def run_strategy(raw_by_track, strategy_fn, score_idx=1):
    """Apply a strategy to all tracks, return {(fidx, tid): score}."""
    out = {}
    for tid, items in raw_by_track.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        flags = strategy_fn(items_sorted)
        flags = smooth_bools(flags)
        for item, flag in zip(items_sorted, flags):
            if flag:
                fidx = item[0]
                score = item[score_idx] if len(item) > score_idx else 0.0
                out[(fidx, tid)] = round(max(float(score), 0.001), 3)
    return out


def write_brake_json(flags, path):
    out = defaultdict(dict)
    for (fidx, tid), conf in flags.items():
        out[str(fidx)][str(tid)] = conf
    with open(path, "w") as f:
        json.dump(dict(out), f, indent=2)


def process_scene(scene):
    raw_path = os.path.join("output", scene, "strip_scores_raw.json")
    if not os.path.exists(raw_path):
        print(f"[{scene}] no strip_scores_raw.json — run detect_brake_lights_detic.py first")
        return
    with open(raw_path) as f:
        raw = json.load(f)

    print(f"[{scene}] {len(raw)} tracks, {sum(len(v) for v in raw.values())} samples")

    # v5 = v4 (delta-hold on strip-red) + motion filter
    v4_flags = run_strategy(raw, strategy_delta_hold, score_idx=1)
    v5_flags = apply_motion_filter(v4_flags, scene)

    # v6 = per-track bulb brightness baseline (no motion filter — works
    # when both ego and target are stopped)
    v6_flags = run_strategy(raw, strategy_bulb_baseline, score_idx=4)

    # v7 = union of v5 and v6
    v7_flags = {}
    for (fidx, tid), conf in v5_flags.items():
        v7_flags[(fidx, tid)] = conf
    for (fidx, tid), conf in v6_flags.items():
        existing = v7_flags.get((fidx, tid), 0.0)
        if conf > existing:
            v7_flags[(fidx, tid)] = conf

    out_path = os.path.join("output", scene, "brake_lights_detic.json")
    write_brake_json(v7_flags, out_path)
    print(f"[{scene}] {len(v7_flags)} brake events -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    args = parser.parse_args()
    if args.scene == "all":
        scenes = sorted(d for d in os.listdir("output")
                        if d.startswith("scene")
                        and os.path.isfile(os.path.join("output", d, "strip_scores_raw.json")))
    else:
        scenes = [args.scene]
    for s in scenes:
        process_scene(s)
