import os
import json
import argparse
import cv2
import math
import numpy as np
import torch
from copy import deepcopy

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type

MM3D_PKG = "/home/shibuwu/miniconda3/envs/mm3d/lib/python3.11/site-packages/mmdet3d/.mim"
CONFIG = os.path.join(MM3D_PKG, "configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py")
WEIGHTS = "models/fcos3d/fcos3d_r101_finetune.pth"
CALIB_PATH = "P3Data/Calib/calibration_results.json"
OUTPUT_DIR = "output"

NUS_CLASSES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
NUS_TO_LABEL = {
    'car': 'vehicle', 'truck': 'vehicle', 'trailer': 'vehicle',
    'bus': 'vehicle', 'construction_vehicle': 'vehicle',
    'bicycle': 'vehicle', 'motorcycle': 'vehicle',
    'pedestrian': 'pedestrian', 'traffic_cone': 'traffic_cone',
    'barrier': 'barrier',
}
NUS_TO_VTYPE = {
    'car': 'sedan',
    'truck': 'truck',
    'trailer': 'truck',
    'bus': 'truck',
    'construction_vehicle': 'truck',
    'bicycle': 'bicycle',
    'motorcycle': 'motorcycle',
}


def load_model(device='cuda:0'):
    cfg = Config.fromfile(CONFIG)
    cfg.model.backbone.init_cfg = None
    model = init_model(cfg, WEIGHTS, device=device)
    model.eval()
    return model


def run_fcos3d(model, img_path, cam2img, device='cuda:0'):
    """Run FCOS3D on a single image."""
    cfg = model.cfg
    test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    data_ = dict(
        images={'CAM_FRONT': {
            'img_path': img_path,
            'cam2img': cam2img.tolist(),
        }},
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
    )
    data_ = test_pipeline(data_)
    collate_data = pseudo_collate([data_])

    with torch.no_grad():
        results = model.test_step(collate_data)
    return results[0]


def get_2d_bbox_from_3d(corners_3d, K, img_w, img_h):
    """Project 3D corners to image and get enclosing 2D bbox."""
    if np.any(corners_3d[:, 2] <= 0.1):
        return None
    pts_2d_h = (K @ corners_3d.T).T
    pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]

    x1 = max(0, np.min(pts_2d[:, 0]))
    y1 = max(0, np.min(pts_2d[:, 1]))
    x2 = min(img_w, np.max(pts_2d[:, 0]))
    y2 = min(img_h, np.max(pts_2d[:, 1]))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def fcos3d_yaw_to_heading(yaw_rad):
    """Convert FCOS3D yaw (camera coords, rotation around Y-down) to our convention.

    FCOS3D: 0=right(+X), 90°=toward(-Z), 180°=left(-X), 270°=away(+Z)
    Ours:   0=away, pi/2=right, pi=toward, -pi/2=left

    Conversion: heading = yaw - 3*pi/2, normalized to [-pi, pi]
    """
    heading = yaw_rad - 3 * math.pi / 2
    # normalize to [-pi, pi]
    while heading > math.pi:
        heading -= 2 * math.pi
    while heading < -math.pi:
        heading += 2 * math.pi
    return heading


def process_scene(model, scene, cam2img, device='cuda:0', score_thr=0.25,
                  step=1):
    det_path = os.path.join(OUTPUT_DIR, scene, "detections.json")
    if not os.path.exists(det_path):
        print(f"    skip: no detections.json")
        return

    with open(det_path) as f:
        frames_data = json.load(f)

    # find front camera video
    undist = f"P3Data/Sequences/{scene}/Undist"
    if not os.path.exists(undist):
        print(f"    skip: no Undist dir")
        return
    vids = [f for f in os.listdir(undist) if 'front' in f.lower()]
    if not vids:
        print(f"    skip: no front video")
        return
    cap = cv2.VideoCapture(os.path.join(undist, vids[0]))

    K = cam2img[:3, :3]
    tmp_path = os.path.join(OUTPUT_DIR, scene, "_tmp_fcos3d.png")
    total_matched, total_dets = 0, 0

    # get unique frame indices with front detections (vehicles + pedestrians)
    FCOS3D_LABELS = {"vehicle", "pedestrian"}
    frame_indices = set()
    for ei, e in enumerate(frames_data):
        if step > 1 and ei % step != 0:
            continue
        for d in e.get("detections", []):
            if d["label"] in FCOS3D_LABELS and d.get("camera") == "front":
                frame_indices.add(e["frame_idx"])
                break

    # process each frame
    for e in frames_data:
        fidx = e["frame_idx"]
        if fidx not in frame_indices:
            continue

        front_dets = [d for d in e.get("detections", [])
                      if d["label"] in FCOS3D_LABELS and d.get("camera") == "front"]
        if not front_dets:
            continue

        # read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imwrite(tmp_path, frame)

        # run FCOS3D
        result = run_fcos3d(model, os.path.abspath(tmp_path), cam2img, device)
        pred = result.pred_instances_3d
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy()
        bboxes_3d = pred.bboxes_3d

        # filter by score
        mask = scores > score_thr
        if mask.sum() == 0:
            continue

        keep = np.where(mask)[0]
        corners_3d = bboxes_3d[keep].corners.cpu().numpy()

        # build FCOS3D 2D bboxes for matching
        fcos_dets = []
        for j, ki in enumerate(keep):
            bbox_2d = get_2d_bbox_from_3d(corners_3d[j], K, 1280, 960)
            if bbox_2d is None:
                continue
            box_tensor = bboxes_3d.tensor[ki].cpu().numpy()
            nus_cls = NUS_CLASSES[labels[ki]]
            our_label = NUS_TO_LABEL.get(nus_cls, 'other')
            fcos_dets.append({
                'bbox_2d': bbox_2d,
                'score': float(scores[ki]),
                'nus_class': nus_cls,
                'our_label': our_label,
                'yaw': float(box_tensor[6]),
                'depth': float(np.sqrt(box_tensor[0]**2 + box_tensor[1]**2 + box_tensor[2]**2)),
                'dims': [float(box_tensor[3]), float(box_tensor[4]), float(box_tensor[5])],
                'pos': [float(box_tensor[0]), float(box_tensor[1]), float(box_tensor[2])],
            })

        # match FCOS3D detections to YOLO detections by IoU
        for det in front_dets:
            yolo_bbox = det["bbox"]
            best_iou, best_fd = 0, None
            for fd in fcos_dets:
                if fd['our_label'] != det['label']:
                    continue
                iou = compute_iou(yolo_bbox, fd['bbox_2d'])
                if iou > best_iou:
                    best_iou = iou
                    best_fd = fd

            if best_fd and best_iou > 0.15:
                heading = fcos3d_yaw_to_heading(best_fd['yaw'])
                det['fcos3d_yaw'] = best_fd['yaw']
                det['fcos3d_heading'] = heading
                det['fcos3d_depth'] = best_fd['depth']
                det['fcos3d_dims'] = best_fd['dims']
                det['fcos3d_pos'] = best_fd['pos']
                det['fcos3d_score'] = best_fd['score']
                det['fcos3d_class'] = best_fd['nus_class']
                det['fcos3d_iou'] = best_iou
                # set vehicle_type for bicycle/motorcycle from nuScenes
                if best_fd['nus_class'] == 'bicycle':
                    det['vehicle_type'] = 'bicycle'
                elif best_fd['nus_class'] == 'motorcycle':
                    det['vehicle_type'] = 'motorcycle'
                total_matched += 1
            total_dets += 1

    cap.release()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with open(det_path, "w") as f:
        json.dump(frames_data, f)
    print(f"    matched {total_matched}/{total_dets} detections")


def main():
    parser = argparse.ArgumentParser(description="FCOS3D 3D detection → detections.json")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.25)
    parser.add_argument("--step", type=int, default=1,
                        help="Only process every Nth frame entry (saves GPU time)")
    args = parser.parse_args()

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted([
            d for d in os.listdir(OUTPUT_DIR)
            if os.path.isdir(os.path.join(OUTPUT_DIR, d))
            and os.path.exists(os.path.join(OUTPUT_DIR, d, "detections.json"))
        ])

    calib = json.load(open(CALIB_PATH))
    cam2img = np.eye(4, dtype=np.float32)
    cam2img[:3, :3] = np.array(calib['front']['K'], dtype=np.float32)

    print("Loading FCOS3D model...")
    model = load_model(args.device)
    print(f"Processing {len(scenes)} scenes...")

    for scene in scenes:
        print(f"  -> {scene}")
        process_scene(model, scene, cam2img, args.device, args.score_thr,
                      step=args.step)

    print("Done.")


if __name__ == "__main__":
    main()
