import sys, os, json, argparse, glob
import numpy as np
import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAFT', 'core'))
from raft import RAFT
from utils.utils import InputPadder

def load_image(path):
    img = cv2.imread(path).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')

def compute_flow(model, frame1_path, frame2_path):
    img1 = load_image(frame1_path)
    img2 = load_image(frame2_path)
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    _, flow_up = model(img1, img2, iters=20, test_mode=True)
    flow_up = padder.unpad(flow_up)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()

def compute_residual_flow(flow, sample_step=10):
    """Compute residual flow after removing ego-motion via homography."""
    H, W = flow.shape[:2]
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    pts1 = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
    pts2 = pts1 + flow

    pts1_s = pts1[::sample_step, ::sample_step].reshape(-1, 2)
    pts2_s = pts2[::sample_step, ::sample_step].reshape(-1, 2)
    H_mat, _ = cv2.findHomography(pts1_s, pts2_s, cv2.RANSAC, 3.0)
    if H_mat is None:
        return np.zeros((H, W)), np.zeros((H, W, 2))

    pts1_flat = pts1.reshape(-1, 2).astype(np.float64)
    ones = np.ones((H * W, 1))
    pts1_h = np.hstack([pts1_flat, ones])
    projected = (H_mat @ pts1_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    expected_flow = (projected - pts1_flat).reshape(H, W, 2)

    residual = flow - expected_flow
    residual_mag = np.sqrt(residual[..., 0]**2 + residual[..., 1]**2)
    return residual_mag, residual

def get_lane_divider_x(frame_data):
    """Find the median x-pixel of solid yellow lane divider."""
    if 'lanes' not in frame_data:
        return None
    # Find leftmost solid lane as the divider (works for yellow or white)
    solid_lanes = [l for l in frame_data['lanes'] if l.get('type') == 'solid']
    if not solid_lanes:
        return None
    # For each solid lane, get its median x position
    lane_medians = []
    for lane in solid_lanes:
        pts_x = [pt[0] for pt in lane.get('points', [])]
        if pts_x:
            lane_medians.append(float(np.median(pts_x)))
    if not lane_medians:
        return None
    # Priority 1: yellow solid lane = center divider
    yellow_xs = []
    for lane in frame_data['lanes']:
        if lane.get('color') == 'yellow':
            for pt in lane.get('points', []):
                yellow_xs.append(pt[0])
    if yellow_xs:
        return float(np.median(yellow_xs))
    
    # Priority 2: no yellow — use leftmost dashed lane as divider
    # (solid white on far left is usually road shoulder, not center)
    dashed_meds = []
    for lane in frame_data['lanes']:
        if lane.get('type') == 'dashed':
            pts_x = [pt[0] for pt in lane.get('points', [])]
            if pts_x:
                dashed_meds.append(float(np.median(pts_x)))
    if dashed_meds:
        return min(dashed_meds)  # leftmost dashed = boundary between oncoming and ego lanes
    
    # Priority 3: use leftmost solid
    if lane_medians:
        return min(lane_medians)
    
    return None

def classify_direction(vx, vy, veh_cx, lane_divider_x):
    """Classify vehicle direction using physics + lane geometry.
    
    Priority:
    1. Lane position (most reliable): left of yellow line = oncoming
    2. Strong flow signal: high |vy| gives clear direction
    3. Default: ahead
    """
    # Primary: lane-based classification (overrides everything)
    if lane_divider_x is not None:
        if veh_cx < lane_divider_x + 70:  # 30px margin for borderline cases
            return 'oncoming'
        else:
            return 'ahead'
    
    # Secondary: flow-based (only when no lane data)
    if vy > 0.02:
        return 'oncoming'
    elif vy < -0.02:
        return 'ahead'
    elif abs(vx) > abs(vy) * 3 and abs(vx) > 0.02:
        return 'left' if vx < 0 else 'right'
    
    return 'ahead'

def classify_vehicles(detections_json, scene_dir, model_path, threshold=0.2):
    # Load RAFT
    args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model = model.module.to('cuda').eval()

    # Find front video
    front_vid = glob.glob(os.path.join(scene_dir, 'Undist', '*front_undistort.mp4'))[0]
    cap = cv2.VideoCapture(front_vid)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    data = json.load(open(detections_json))

    # Load camera intrinsics
    calib_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'P3Data', 'Calib', 'calibration_results.json')
    cam_K = None
    if os.path.exists(calib_file):
        calib_data = json.load(open(calib_file))
        if 'front' in calib_data:
            cam_K = np.array(calib_data['front']['K'])
            print(f'Loaded intrinsics: fx={cam_K[0,0]:.1f} fy={cam_K[1,1]:.1f}')
    if cam_K is None:
        print('WARN: no calibration — using flow direction only')

    with torch.no_grad():
        for frame_data in data:
            fidx = frame_data['frame_idx']
            vehicles = [d for d in frame_data['detections'] if d['label'] == 'vehicle']
            if not vehicles:
                continue

            if fidx + 1 >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret1, img1 = cap.read()
            ret2, img2 = cap.read()
            if not ret1 or not ret2:
                continue

            cv2.imwrite('/tmp/_f1.png', img1)
            cv2.imwrite('/tmp/_f2.png', img2)

            flow = compute_flow(model, '/tmp/_f1.png', '/tmp/_f2.png')
            residual_mag, residual = compute_residual_flow(flow)
            H, W = residual_mag.shape

            # Get lane divider position for this frame
            lane_divider_x = get_lane_divider_x(frame_data)
            if lane_divider_x is not None:
                print(f'  Frame {fidx}: lane divider at x={lane_divider_x:.0f}')

            for d in vehicles:
                x1, y1, x2, y2 = [int(v) for v in d['bbox']]
                x1, y1, x2, y2 = max(0,x1), max(0,y1), min(W,x2), min(H,y2)
                roi_mag = residual_mag[y1:y2, x1:x2]
                med = float(np.median(roi_mag))
                moving = med > threshold
                d['moving'] = moving
                d['residual_score'] = round(med, 3)

                if moving:
                    # Raw flow for display
                    roi_raw = flow[y1:y2, x1:x2]
                    mf = np.mean(roi_raw, axis=(0, 1))
                    d['flow_direction'] = [round(float(mf[0]), 2), round(float(mf[1]), 2)]

                    # Residual flow for direction
                    res_roi = residual[y1:y2, x1:x2]
                    res_mean = np.mean(res_roi, axis=(0, 1))
                    dx_img, dy_img = res_mean[0], res_mean[1]

                    # Backproject to 3D velocity
                    bbox_h = y2 - y1
                    if cam_K is not None and bbox_h > 10:
                        fx, fy = cam_K[0, 0], cam_K[1, 1]
                        Z = fx * 1.5 / bbox_h
                        vx = dx_img * Z / fx
                        vy = dy_img * Z / fy
                    else:
                        vx, vy, Z = float(dx_img), float(dy_img), -1.0

                    d['velocity_3d'] = [round(float(vx), 4), round(float(vy), 4)]
                    d['depth_Z'] = round(float(Z), 2)

                    # Classify direction
                    veh_cx = (x1 + x2) / 2
                    d['move_dir'] = classify_direction(vx, vy, veh_cx, lane_divider_x)

                    print(f'  f{fidx} [{x1},{y1},{x2},{y2}] Z={Z:.1f} vx={vx:.4f} vy={vy:.4f} cx={veh_cx:.0f} -> {d["move_dir"]}')
                else:
                    d['flow_direction'] = [0.0, 0.0]
                    d['move_dir'] = 'parked'

            n_mov = sum(1 for d in vehicles if d.get('moving'))
            n_par = sum(1 for d in vehicles if not d.get('moving'))
            print(f"Frame {fidx}: {n_mov} moving, {n_par} parked")

    cap.release()
    json.dump(data, open(detections_json, 'w'), indent=2)
    print(f"Updated {detections_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--model', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RAFT', 'models', 'raft-things.pth'))
    parser.add_argument('--threshold', type=float, default=0.2)
    args = parser.parse_args()

    scene_dir = os.path.expanduser(f'~/CV/proj_3/P3Data/Sequences/{args.scene}')
    det_json = os.path.expanduser(f'~/CV/proj_3/output/{args.scene}/detections.json')
    classify_vehicles(det_json, scene_dir, args.model, args.threshold)
