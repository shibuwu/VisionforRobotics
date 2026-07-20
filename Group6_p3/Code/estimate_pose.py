import os
import sys
import json
import argparse
import cv2
import numpy as np

# RTMPose3D project needs to be importable
MMPOSE_DIR = os.path.join(os.path.dirname(__file__), "models", "mmpose")
RTMPOSE3D_DIR = os.path.join(MMPOSE_DIR, "projects", "rtmpose3d")
sys.path.insert(0, RTMPOSE3D_DIR)

# config
DET_CONFIG = os.path.join(RTMPOSE3D_DIR, "demo", "rtmdet_m_640-8xb32_coco-person.py")
DET_WEIGHTS = os.path.join(MMPOSE_DIR, "checkpoints", "rtmdet_m_person.pth")

POSE_CONFIG = os.path.join(RTMPOSE3D_DIR, "configs", "rtmw3d-l_8xb64_cocktail14-384x288.py")
POSE_WEIGHTS = os.path.join(MMPOSE_DIR, "checkpoints", "rtmw3d-l_cocktail14.pth")

DATA_DIR = "P3Data"
SEQ_DIR = os.path.join(DATA_DIR, "Sequences")
OUTPUT_DIR = "output"

CAM_VIDEO_KEYS = {
    "front": "front",
    "back": "back",
    "left": "left_repeater",
    "right": "right_repeater",
}

IOU_THRESHOLD = 0.2
DET_BBOX_THR = 0.5


# helpers
def find_video(undist_dir, cam_name):
    key = CAM_VIDEO_KEYS[cam_name]
    for f in os.listdir(undist_dir):
        if key in f and f.endswith(".mp4"):
            return os.path.join(undist_dir, f)
    return None


def iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def load_models(device):
    """Load RTMDet person detector and RTMPose3D estimator."""
    from mmdet.apis import init_detector
    from mmpose.apis import init_model
    from mmpose.utils import adapt_mmdet_pipeline

    # import RTMPose3D custom modules so they register with mmpose
    import importlib
    rtmpose3d_pkg = importlib.import_module("rtmpose3d")
    # trigger __init__ which registers custom modules
    for attr in dir(rtmpose3d_pkg):
        getattr(rtmpose3d_pkg, attr)

    detector = init_detector(DET_CONFIG, DET_WEIGHTS, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    pose_model = init_model(POSE_CONFIG, POSE_WEIGHTS, device=device)

    return detector, pose_model


def run_pose_on_frame(detector, pose_model, frame, device):
    """Run RTMPose3D on a frame. Returns (bboxes, keypoints_2d, keypoints_3d, scores).

    keypoints_2d: (N, 133, 2) pixel coords
    keypoints_3d: (N, 133, 3) relative 3D coords
    """
    from mmdet.apis import inference_detector
    from mmpose.apis import inference_topdown

    # detect persons
    det_result = inference_detector(detector, frame)
    pred = det_result.pred_instances.cpu().numpy()
    bboxes = pred.bboxes[np.logical_and(
        pred.labels == 0, pred.scores > DET_BBOX_THR)]

    if len(bboxes) == 0:
        return np.zeros((0, 4)), np.zeros((0, 17, 2)), np.zeros((0, 17, 3)), np.zeros((0, 17))

    # run 3D pose
    pose_results = inference_topdown(pose_model, frame, bboxes)

    all_bboxes = []
    all_kpts_2d = []
    all_kpts_3d = []
    all_scores = []

    for result in pose_results:
        pred_inst = result.pred_instances
        kpts = pred_inst.keypoints  # (1, 133, 3) for 3D or (1, 133, 2) for 2D
        scores = pred_inst.keypoint_scores  # (1, 133) or (1, 1, 133)

        if scores.ndim == 3:
            scores = np.squeeze(scores, axis=1)
        if kpts.ndim == 4:
            kpts = np.squeeze(kpts, axis=1)

        # kpts is 3D: (1, 133, 3) — transform to standard coords
        kpts_3d = kpts.copy()
        # RTMPose3D convention: negate and swap axes → (x_right, z_forward, y_up)
        kpts_3d = -kpts_3d[..., [0, 2, 1]]
        # rebase so lowest point is at z=0
        kpts_3d[..., 2] -= np.min(kpts_3d[..., 2], axis=-1, keepdims=True)

        # 2D pixel coordinates from transformed_keypoints
        kpts_2d = pred_inst.transformed_keypoints  # (1, 133, 2) in pixel coords
        if kpts_2d.ndim == 4:
            kpts_2d = np.squeeze(kpts_2d, axis=1)

        # only keep first 17 (COCO body) for compatibility, but store all 133
        all_bboxes.append(result.pred_instances.bboxes[0] if hasattr(result.pred_instances, 'bboxes') else bboxes[0])
        all_kpts_2d.append(kpts_2d[0])     # (133, 2)
        all_kpts_3d.append(kpts_3d[0])     # (133, 3)
        all_scores.append(scores[0])       # (133,)

    return (np.array(all_bboxes) if all_bboxes else np.zeros((0, 4)),
            np.array(all_kpts_2d),
            np.array(all_kpts_3d),
            np.array(all_scores))


# main pipeline
def process_scene(scene, detector, pose_model, device, step=1):
    det_path = os.path.join(OUTPUT_DIR, scene, "detections.json")
    if not os.path.exists(det_path):
        print(f"    skip: no detections.json")
        return

    with open(det_path) as f:
        frames_data = json.load(f)

    # collect which (frame_idx, camera) combos have pedestrians
    # subsample: only process every Nth detection entry
    ped_frames = {}
    for ei, entry in enumerate(frames_data):
        if step > 1 and ei % step != 0:
            continue
        fidx = entry["frame_idx"]
        for di, det in enumerate(entry.get("detections", [])):
            if det["label"] == "pedestrian":
                cam = det.get("camera", "front")
                key = (fidx, cam)
                ped_frames.setdefault(key, []).append((ei, di))

    if not ped_frames:
        print(f"    no pedestrians found")
        return

    cams_needed = set(cam for (_, cam) in ped_frames.keys())
    undist_dir = os.path.join(SEQ_DIR, scene, "Undist")

    total, matched = 0, 0

    for cam in cams_needed:
        video_path = find_video(undist_dir, cam)
        if video_path is None:
            print(f"    warning: no video for camera {cam}")
            continue

        cam_frame_idxs = sorted(set(
            fidx for (fidx, c) in ped_frames.keys() if c == cam
        ))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    warning: cannot open {video_path}")
            continue

        current_frame = -1

        for target_fidx in cam_frame_idxs:
            if target_fidx != current_frame + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_fidx)
            ret, frame = cap.read()
            current_frame = target_fidx
            if not ret:
                continue

            # run RTMPose3D
            pose_bboxes, kpts_2d, kpts_3d, kpt_scores = run_pose_on_frame(
                detector, pose_model, frame, device)

            if len(pose_bboxes) == 0:
                continue

            # match to our pedestrian detections
            key = (target_fidx, cam)
            for ei, di in ped_frames[key]:
                det = frames_data[ei]["detections"][di]
                total += 1

                best_iou, best_pi = 0, -1
                for pi in range(len(pose_bboxes)):
                    v = iou(det["bbox"], pose_bboxes[pi].tolist())
                    if v > best_iou:
                        best_iou = v
                        best_pi = pi

                if best_iou >= IOU_THRESHOLD and best_pi >= 0:
                    # store first 17 COCO body keypoints as [x, y, conf]
                    kp2d = kpts_2d[best_pi][:17]  # (17, 2)
                    scores = kpt_scores[best_pi][:17]  # (17,)
                    det["keypoints"] = [[float(kp2d[i][0]), float(kp2d[i][1]), float(scores[i])] for i in range(17)]
                    # store 3D keypoints (first 17 body joints)
                    kp3d = kpts_3d[best_pi][:17]  # (17, 3)
                    det["keypoints_3d"] = [[float(kp3d[i][0]), float(kp3d[i][1]), float(kp3d[i][2])] for i in range(17)]
                    matched += 1
                else:
                    # fallback: nearest by bbox center
                    dcx = (det["bbox"][0] + det["bbox"][2]) / 2
                    dcy = (det["bbox"][1] + det["bbox"][3]) / 2
                    best_dist, best_pi2 = float('inf'), -1
                    for pi in range(len(pose_bboxes)):
                        pcx = (pose_bboxes[pi][0] + pose_bboxes[pi][2]) / 2
                        pcy = (pose_bboxes[pi][1] + pose_bboxes[pi][3]) / 2
                        dist = ((dcx - pcx)**2 + (dcy - pcy)**2)**0.5
                        if dist < best_dist:
                            best_dist, best_pi2 = dist, pi
                    if best_pi2 >= 0 and best_dist < 100:
                        kp2d = kpts_2d[best_pi2][:17]
                        scores = kpt_scores[best_pi2][:17]
                        det["keypoints"] = [[float(kp2d[i][0]), float(kp2d[i][1]), float(scores[i])] for i in range(17)]
                        kp3d = kpts_3d[best_pi2][:17]
                        det["keypoints_3d"] = [[float(kp3d[i][0]), float(kp3d[i][1]), float(kp3d[i][2])] for i in range(17)]
                        matched += 1

        cap.release()

    with open(det_path, "w") as f:
        json.dump(frames_data, f)
    print(f"    {matched}/{total} pedestrians got 3D keypoints")


def main():
    parser = argparse.ArgumentParser(description="Pedestrian 3D pose (RTMPose3D)")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if args.device != "cpu" else "cpu"
    print(f"Device: {device}")

    print("Loading RTMDet + RTMPose3D...")
    detector, pose_model = load_models(device)

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
        print(f"  → {scene}")
        process_scene(scene, detector, pose_model, device, step=args.step)

    print("Done.")


if __name__ == "__main__":
    main()
