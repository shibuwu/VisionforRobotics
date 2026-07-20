import os
import json
import argparse
import cv2
from ultralytics import YOLO

OUTPUT_DIR = "output"
MIN_CROP_PX = 20

CALIB_FILE = "P3Data/Calib/calibration_results.json"
_FY_FRONT = None

def get_fy_front():
    global _FY_FRONT
    if _FY_FRONT is None:
        with open(CALIB_FILE) as f:
            calib = json.load(f)
        _FY_FRONT = float(calib["front"]["K"][1][1])
    return _FY_FRONT


def height_fallback(det):
    """Estimate vehicle asset from bbox pixel height + depth.
    Returns (asset_type, conf) or (None, 0) if depth missing.
    H_real = (v_max - v_min) * Z / fy"""
    bbox = det["bbox"]
    bh_px = bbox[3] - bbox[1]
    z = det.get("fcos3d_depth") or det.get("depth")
    if not z or z <= 0 or bh_px < 5:
        return None, 0.0
    fy = get_fy_front()
    h_real = bh_px * z / fy
    if h_real < 1.55:
        return "sedan", 0.5
    elif h_real < 2.0:
        return "suv", 0.5
    else:
        return "truck", 0.5

IMAGENET_TO_ASSET = {
    436: "sedan",   # beach wagon (station wagon — closest to sedan family)
    468: "sedan",   # cab, taxi
    511: "sedan",   # convertible
    609: "sedan",   # jeep, landrover  -> actually SUV; overridden below
    627: "sedan",   # limousine
    656: "suv",     # minivan
    661: "sedan",   # Model T
    717: "pickup",  # pickup, pickup truck
    751: "sedan",   # racer, race car
    817: "sedan",   # sports car
    864: "sedan",   # tow truck (small) — could be either, default sedan
    867: "sedan",   # trailer truck — actually big truck, overridden below
    874: "sedan",   # trolleybus
    407: "sedan",   # ambulance — van-ish
    555: "sedan",   # fire engine
    569: "sedan",   # garbage truck
}
# corrections — overriding the rough table above
IMAGENET_TO_ASSET[609] = "suv"     # jeep, landrover
IMAGENET_TO_ASSET[867] = "sedan"   # trailer truck — but YOLO already tags these as "truck"

VEHICLE_CLS_IDS = set(IMAGENET_TO_ASSET.keys())


def crop_bbox(frame, bbox, pad_ratio=0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px))
    y2 = min(h, int(y2 + py))
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < MIN_CROP_PX or crop.shape[1] < MIN_CROP_PX:
        return None
    return crop


def load_frame(scene, fidx):
    frames_dir = os.path.join(OUTPUT_DIR, scene, "frames")
    for ext in (".jpg", ".png"):
        p = os.path.join(frames_dir, f"frame_{fidx:05d}{ext}")
        if os.path.exists(p):
            return cv2.imread(p)
    return None


def classify_crop(model, crop_bgr):
    """Run YOLO11-cls on a crop, return (asset_type or None, conf, raw_class_name)."""
    res = model.predict(crop_bgr, verbose=False, imgsz=224)[0]
    probs = res.probs
    top5_ids = probs.top5
    top5_conf = probs.top5conf.tolist()
    for cls_id, conf in zip(top5_ids, top5_conf):
        if cls_id in IMAGENET_TO_ASSET:
            return IMAGENET_TO_ASSET[cls_id], float(conf), res.names[cls_id]
    # no vehicle class in top-5 → caller should use height fallback
    top1_id = probs.top1
    return None, float(probs.top1conf), res.names[top1_id]


def process_scene(scene, model):
    src = os.path.join(OUTPUT_DIR, scene, "detections.json")
    dst = os.path.join(OUTPUT_DIR, scene, "detections_imagenet.json")
    if not os.path.exists(src):
        print(f"    skip: no detections.json")
        return

    with open(src) as f:
        data = json.load(f)

    total, classified, skipped, fallback = 0, 0, 0, 0
    frame_cache = {}

    for entry in data:
        fidx = entry["frame_idx"]
        for det in entry.get("detections", []):
            if det.get("label") != "vehicle":
                continue
            total += 1
            yolo_type = det.get("vehicle_type", "")
            if yolo_type in ("bicycle", "motorcycle", "truck"):
                det["vehicle_color"] = "offwhite"
                skipped += 1
                continue

            if fidx not in frame_cache:
                frame_cache[fidx] = load_frame(scene, fidx)
            frame = frame_cache[fidx]
            if frame is None:
                det["vehicle_type"] = "sedan"
                det["vehicle_color"] = "offwhite"
                continue

            crop = crop_bbox(frame, det["bbox"])
            if crop is None:
                det["vehicle_type"] = "sedan"
                det["vehicle_color"] = "offwhite"
                continue

            vt, conf, raw = classify_crop(model, crop)
            if vt is None:
                # ImageNet didn't see a vehicle in top-5; try height fallback
                fb_vt, fb_conf = height_fallback(det)
                if fb_vt is not None:
                    vt, conf = fb_vt, fb_conf
                    raw = f"height_fallback={raw}"
                    fallback += 1
                else:
                    vt = "sedan"
            det["vehicle_type"] = vt
            det["vtype_confidence"] = conf
            det["vehicle_color"] = "offwhite"
            det["imagenet_raw"] = raw
            classified += 1

        if len(frame_cache) > 50:
            frame_cache.clear()

    with open(dst, "w") as f:
        json.dump(data, f)
    print(f"    {classified}/{total} classified ({fallback} via height fallback), {skipped} skipped (yolo bike/moto/truck) -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Vehicle sub-classification (YOLO11n-cls ImageNet)")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--model", default="yolo11m-cls.pt")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = YOLO(args.model)
    print(f"  ImageNet classes mapped to assets: {len(IMAGENET_TO_ASSET)}")

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([
            d for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))
            and os.path.exists(os.path.join(OUTPUT_DIR, d, "detections.json"))
        ])

    print(f"Processing {len(scenes)} scenes...")
    for s in scenes:
        print(f"  -> {s}")
        process_scene(s, model)
    print("Done.")


if __name__ == "__main__":
    main()
