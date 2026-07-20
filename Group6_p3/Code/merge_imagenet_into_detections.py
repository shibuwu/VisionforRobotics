import json
import os
import shutil
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENES_DEFAULT = ["scene3", "scene5", "scene6", "scene7", "scene8", "scene13"]


def merge_one(scene):
    out_dir = os.path.join(PROJECT_DIR, "output", scene)
    main_path = os.path.join(out_dir, "detections.json")
    imnet_path = os.path.join(out_dir, "detections_imagenet.json")
    backup_path = os.path.join(out_dir, "detections_pre_imagenet.json")

    if not os.path.exists(main_path):
        print(f"[{scene}] no detections.json, skipping")
        return
    if not os.path.exists(imnet_path):
        print(f"[{scene}] no detections_imagenet.json, skipping")
        return

    with open(main_path) as f:
        main = json.load(f)
    with open(imnet_path) as f:
        imnet = json.load(f)

    if not os.path.exists(backup_path):
        shutil.copy(main_path, backup_path)
        print(f"[{scene}] backed up to {os.path.basename(backup_path)}")

    imnet_by_frame = {e["frame_idx"]: e for e in imnet}
    patched = 0
    for entry in main:
        ie = imnet_by_frame.get(entry["frame_idx"])
        if ie is None:
            continue
        for det, idet in zip(entry.get("detections", []),
                             ie.get("detections", [])):
            if det.get("label") != "vehicle":
                continue
            if det.get("bbox") != idet.get("bbox"):
                continue
            for k in ("vehicle_type", "vtype_confidence", "imagenet_raw"):
                if k in idet:
                    det[k] = idet[k]
            patched += 1

    with open(main_path, "w") as f:
        json.dump(main, f)
    print(f"[{scene}] merged {patched} vehicle subtypes into detections.json")


def main():
    scenes = sys.argv[1:] or SCENES_DEFAULT
    for s in scenes:
        merge_one(s)


if __name__ == "__main__":
    main()
