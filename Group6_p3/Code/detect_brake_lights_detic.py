import argparse
import cv2
import json
import numpy as np
import os
import sys
from collections import defaultdict


DETIC_DIR = os.environ.get(
    "DETIC_DIR",
    os.path.expanduser("~/Desktop/drspring/realrobot/Detic"))
sys.path.insert(0, DETIC_DIR)
sys.path.insert(0, os.path.join(DETIC_DIR, "third_party/CenterNet2"))

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

DETIC_CONFIG = os.path.join(
    DETIC_DIR,
    "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
DETIC_WEIGHTS = os.path.join(
    DETIC_DIR,
    "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")

LVIS_TAILLIGHT = 1055
LVIS_BRAKE_LIGHT = 1019
LVIS_BLINKER = 113
LVIS_HEADLIGHT = 548   # filtered out — front of car
LVIS_REFLECTOR = 879   # filtered out — non-lit passive element

REAR_LIGHT_IDS = {LVIS_TAILLIGHT, LVIS_BRAKE_LIGHT, LVIS_BLINKER}
SKIP_IDS = {LVIS_HEADLIGHT, LVIS_REFLECTOR}

PARTS_VOCAB = {
    LVIS_TAILLIGHT:   "taillight",
    LVIS_BRAKE_LIGHT: "brake_light",
    LVIS_BLINKER:     "blinker",
    LVIS_HEADLIGHT:   "headlight",
    LVIS_REFLECTOR:   "reflector",
}

OUTPUT_DIR = "output"
DETIC_CONF = 0.40               
BRAKE_MASKED_FRAC = 0.025       
AMBER_FRAC_MIN = 0.25           
FLASH_WINDOW = 30              
FLASH_MIN_TRANSITIONS = 5       
FLASH_MIN_ON_FRAMES = 2         
BRAKE_SMOOTH_WINDOW = 3
BRAKE_SMOOTH_MIN_VOTES = 2


def build_detic(device="cuda"):
    """Build Detic with its default LVIS classifier head — no CLIP prompts.
    LVIS already has `taillight`, `brake_light`, `blinker`, `headlight` as
    real classes (ids 1056, 1020, 114, 549) so we just filter the output."""
    orig_cwd = os.getcwd()
    os.chdir(DETIC_DIR)

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(DETIC_CONFIG)
    cfg.MODEL.WEIGHTS = DETIC_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETIC_CONF
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = (
        "datasets/metadata/lvis-21k_clip_a+cname.npy")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20352
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.MODEL.DEVICE = device
    predictor = DefaultPredictor(cfg)

    os.chdir(orig_cwd)
    return predictor


def detic_on_frame(predictor, frame):
    """Return list of (bbox, lvis_cls_id, score) for rear-light detections
    only (taillight, brake_light, blinker — skips everything else)."""
    outputs = predictor(frame)
    inst = outputs["instances"].to("cpu")
    out = []
    for i in range(len(inst)):
        cls_id = int(inst.pred_classes[i].item())
        if cls_id not in REAR_LIGHT_IDS:
            continue  # not a rear light — skip
        bbox = inst.pred_boxes.tensor[i].numpy().tolist()
        score = float(inst.scores[i].item())
        out.append((bbox, cls_id, score))
    return out


# signal helpers
def bright_fraction(frame, bbox):
    """Fraction of pixels inside bbox with V>=200 (grayscale brightness)."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.count_nonzero(hsv[:, :, 2] >= 200)) / hsv[:, :, 2].size


def amber_fraction(frame, bbox):
    """Fraction of pixels that are amber (H 16-30, S>=100, V>=170)."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (H >= 16) & (H <= 30) & (S >= 100) & (V >= 170)
    return float(np.count_nonzero(mask)) / mask.size


def masked_strip_signals(frame, vehicle_bbox, taillight_bboxes):
    """For each Detic taillight bbox, measure red + bright pixel fractions
    in an L-shaped region: a thin horizontal strip extending left/right
    of the taillight AND a box extending DOWNWARD below it.

    Rationale: brake bulbs commonly sit DIRECTLY BELOW the main taillight
    (think rear cluster layouts where the tail is on top and the brake/
    reverse lights are below), or spill into the bumper reflector area.
    Extending the strip downward by ~1.5x the taillight height captures
    that region."""
    if not taillight_bboxes:
        return 0.0, 0.0

    vx1, vy1, vx2, vy2 = vehicle_bbox
    h, w = frame.shape[:2]
    best_red = 0.0
    best_bright = 0.0

    for (tx1, ty1, tx2, ty2) in taillight_bboxes:
        tw = tx2 - tx1
        th = ty2 - ty1
        if tw < 3 or th < 3:
            continue

        # strip region: wider horizontally, tall enough to include area
        # under the taillight where brake/reflector sit
        ext_x = tw * 1.0
        pad_up = th * 0.15
        pad_down = th * 1.5       # extend far down to cover brake/reflector below
        sx1 = max(int(tx1 - ext_x), int(vx1))
        sx2 = min(int(tx2 + ext_x), int(vx2))
        sy1 = max(int(ty1 - pad_up), int(vy1))
        sy2 = min(int(ty2 + pad_down), int(vy2))
        sx1 = max(sx1, 0)
        sy1 = max(sy1, 0)
        sx2 = min(sx2, w)
        sy2 = min(sy2, h)
        if sx2 - sx1 < 5 or sy2 - sy1 < 3:
            continue

        strip = frame[sy1:sy2, sx1:sx2]
        if strip.size == 0:
            continue

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # "warm" = red through orange/yellow. Brake spillover at distance
        # often looks orange-red or even yellowish on dusty lens plastic,
        # so expanding the hue range catches more real brake signal.
        is_warm = (((H <= 30) | (H >= 160)) & (S >= 70) & (V >= 180))
        is_bright = V >= 200

        # mask out the taillight bbox itself
        valid = np.ones_like(is_warm, dtype=bool)
        mx1 = max(int(tx1) - sx1, 0)
        my1 = max(int(ty1) - sy1, 0)
        mx2 = min(int(tx2) - sx1, is_warm.shape[1])
        my2 = min(int(ty2) - sy1, is_warm.shape[0])
        if mx2 > mx1 and my2 > my1:
            valid[my1:my2, mx1:mx2] = False

        denom = int(valid.sum())
        if denom == 0:
            continue
        red_score = float(np.count_nonzero(is_warm & valid)) / denom
        bright_score = float(np.count_nonzero(is_bright & valid)) / denom
        if red_score > best_red:
            best_red = red_score
        if bright_score > best_bright:
            best_bright = bright_score

    return best_red, best_bright


def bbox_contains(outer, inner, margin=5):
    """Is the `inner` bbox mostly inside `outer` (with a small margin)?"""
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return (ix1 >= ox1 - margin and iy1 >= oy1 - margin
            and ix2 <= ox2 + margin and iy2 <= oy2 + margin)


def smooth_bools(flags, window=BRAKE_SMOOTH_WINDOW, min_votes=BRAKE_SMOOTH_MIN_VOTES):
    out = []
    for i in range(len(flags)):
        lo = max(0, i - window // 2)
        hi = min(len(flags), i + window // 2 + 1)
        out.append(sum(flags[lo:hi]) >= min_votes)
    return out


def detect_flash(items, value_idx):
    """items: list of (fidx, red_score, left_amber, right_amber)."""
    n = len(items)
    flags = [False] * n
    if n < 4:
        return flags
    for i in range(n):
        fi = items[i][0]
        window = [items[j][value_idx] for j in range(n)
                  if abs(items[j][0] - fi) <= FLASH_WINDOW]
        if len(window) < 4:
            continue
        binarized = [v > AMBER_FRAC_MIN for v in window]
        if sum(binarized) < FLASH_MIN_ON_FRAMES:
            continue
        trans = sum(1 for j in range(1, len(binarized))
                    if binarized[j] != binarized[j - 1])
        if trans >= FLASH_MIN_TRANSITIONS:
            flags[i] = True
    return flags


# per-scene pipeline
def process_scene(scene_name, predictor, step=1):
    det_file = os.path.join(OUTPUT_DIR, scene_name, "detections.json")
    frames_dir = os.path.join(OUTPUT_DIR, scene_name, "frames")
    brake_out = os.path.join(OUTPUT_DIR, scene_name, "brake_lights_detic.json")
    ind_out = os.path.join(OUTPUT_DIR, scene_name, "indicators_detic.json")
    if not os.path.exists(det_file) or not os.path.exists(frames_dir):
        print(f"[{scene_name}] missing inputs, skipping")
        return

    with open(det_file) as f:
        all_frames = json.load(f)

    # per-track time series: (fidx, brake_score, left_amber_score, right_amber_score)
    by_track = defaultdict(list)
    n_parts_detected = 0

    frames_to_do = [(i, e) for i, e in enumerate(all_frames) if i % step == 0]
    print(f"[{scene_name}] running Detic on {len(frames_to_do)} frames "
          f"(step={step})")

    for ei, entry in frames_to_do:
        fidx = entry["frame_idx"]
        frame_path = os.path.join(frames_dir, f"frame_{fidx:05d}.jpg")
        if not os.path.exists(frame_path):
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # gather tracked front-camera vehicle bboxes
        vehicles = []
        for det in entry.get("detections", []):
            if det.get("label") != "vehicle":
                continue
            if det.get("camera", "front") != "front":
                continue
            tid = det.get("track_id")
            if tid is None:
                continue
            vehicles.append((tid, det["bbox"]))
        if not vehicles:
            continue

        # one Detic pass per frame — finds all light/part boxes in the scene
        parts = detic_on_frame(predictor, frame)
        n_parts_detected += len(parts)

        # for each vehicle, pick the brightest taillight/brake box inside it
        # and measure amber on the left/right half of the same box
        for tid, vbox in vehicles:
            vx1, vy1, vx2, vy2 = vbox
            vmid_x = (vx1 + vx2) / 2.0

            best_brake_score = 0.0
            left_amber_score = 0.0
            right_amber_score = 0.0

            # collect taillight bboxes that belong to this vehicle so we can
            # mask them out of the brake-signal region
            vehicle_taillight_bboxes = []
            for pbox, pcls, pscore in parts:
                if not bbox_contains(vbox, pbox):
                    continue
                if pcls in (LVIS_TAILLIGHT, LVIS_BRAKE_LIGHT):
                    vehicle_taillight_bboxes.append(pbox)

                # amber drives indicator detection — split left/right by midline
                a = amber_fraction(frame, pbox)
                if a > 0:
                    pcx = (pbox[0] + pbox[2]) / 2.0
                    if pcx < vmid_x:
                        left_amber_score = max(left_amber_score, a)
                    else:
                        right_amber_score = max(right_amber_score, a)

            if vehicle_taillight_bboxes:
                strip_red, strip_bright = masked_strip_signals(
                    frame, vbox, vehicle_taillight_bboxes)
                bulb_score = max(bright_fraction(frame, pb)
                                 for pb in vehicle_taillight_bboxes)
            else:
                strip_red = 0.0
                strip_bright = 0.0
                bulb_score = 0.0

            by_track[tid].append(
                (fidx, strip_red, left_amber_score, right_amber_score,
                 bulb_score, strip_bright))

    # classify
    brake_flags = {}   # (fidx, tid) -> confidence float in [0,1]
    left_flags = {}
    right_flags = {}

    for tid, items in by_track.items():
        items.sort()
        n = len(items)

        raw_brake = [item[1] >= BRAKE_MASKED_FRAC for item in items]
        smoothed = smooth_bools(raw_brake)
        confs = [item[1] for item in items]

        lf = detect_flash(items, 2)
        rf = detect_flash(items, 3)

        for idx, item in enumerate(items):
            fidx = item[0]
            if smoothed[idx]:
                brake_flags[(fidx, tid)] = round(float(confs[idx]), 3)
            if lf[idx]:
                left_flags[(fidx, tid)] = True
            if rf[idx]:
                right_flags[(fidx, tid)] = True

    out_brake = defaultdict(dict)
    for (fidx, tid), conf in brake_flags.items():
        out_brake[str(fidx)][str(tid)] = conf
    with open(brake_out, "w") as f:
        json.dump(dict(out_brake), f, indent=2)

    with open(ind_out, "w") as f:
        json.dump({}, f)

    raw_out_file = os.path.join(OUTPUT_DIR, scene_name, "strip_scores_raw.json")
    raw = {}
    for tid, items in by_track.items():
        raw[str(tid)] = [[fidx, round(sr, 4), round(la, 4), round(ra, 4),
                          round(bulb, 4), round(sb, 4)]
                         for (fidx, sr, la, ra, bulb, sb) in items]
    with open(raw_out_file, "w") as f:
        json.dump(raw, f)

    print(f"[{scene_name}] tracks={len(by_track)} "
          f"parts_detected={n_parts_detected} "
          f"brake={len(brake_flags)} "
          f"left_ind={len(left_flags)} right_ind={len(right_flags)}")
    print(f"[{scene_name}] -> {brake_out}, {ind_out}, {raw_out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name or 'all'")
    parser.add_argument("--step", type=int, default=1,
                        help="Run Detic every Nth frame entry")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading Detic...")
    predictor = build_detic(device=args.device)
    print("Detic ready.")

    if args.scene == "all":
        scenes = sorted(d for d in os.listdir(OUTPUT_DIR)
                        if d.startswith("scene")
                        and os.path.isdir(os.path.join(OUTPUT_DIR, d)))
    else:
        scenes = [args.scene]

    for s in scenes:
        process_scene(s, predictor, step=args.step)
