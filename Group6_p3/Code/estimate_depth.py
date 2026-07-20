"""
Metric depth estimation using Metric-Video-Depth-Anything-Large.
Outputs depth in meters. Uses camera focal length for metric scale.
"""

import sys
import os
import numpy as np
import torch
import cv2
import gc
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "Video-Depth-Anything"))

from video_depth_anything.video_depth import VideoDepthAnything

DATA_DIR = "P3Data"
SEQ_DIR = os.path.join(DATA_DIR, "Sequences")
OUTPUT_DIR = "output"
CALIB_FILE = os.path.join(DATA_DIR, "Calib", "calibration_results.json")

# metric model — outputs depth in meters
CHECKPOINT = os.path.expanduser("~/metric_video_depth_anything_vitl.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 20  # small chunks for vitl on 6GB VRAM (fp16)


def load_model():
    config = {'encoder': 'vitl', 'features': 256,
              'out_channels': [256, 512, 1024, 1024], 'metric': True}
    model = VideoDepthAnything(**config)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu'), strict=True)
    model = model.to(DEVICE).eval()
    return model


def read_frames_chunk(cap, start, count, max_res=1280):
    """Read a chunk of frames, resized to max_res."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if max(h, w) > max_res:
            scale = max_res / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def estimate_depth_for_scene(model, scene_name, sample_rate=36):
    """Run metric depth estimation and save depth maps (in meters)."""
    scene_dir = os.path.join(SEQ_DIR, scene_name)
    undist_dir = os.path.join(scene_dir, "Undist")

    front_vid = None
    for f in os.listdir(undist_dir):
        if "front" in f and f.endswith(".mp4"):
            front_vid = os.path.join(undist_dir, f)
            break

    if not front_vid:
        print(f"No front video for {scene_name}")
        return

    cap = cv2.VideoCapture(front_vid)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Running metric depth on {scene_name}...")
    print(f"  {total_frames} frames at {fps:.1f} fps, chunk size {CHUNK_SIZE}")

    depth_dir = os.path.join(OUTPUT_DIR, scene_name, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    overlap = 10
    saved = 0
    chunk_start = 0

    while chunk_start < total_frames:
        chunk_end = min(chunk_start + CHUNK_SIZE, total_frames)
        read_start = max(0, chunk_start - overlap) if chunk_start > 0 else 0
        frames = read_frames_chunk(cap, read_start, chunk_end - read_start)

        if not frames:
            break

        print(f"  Chunk [{chunk_start}-{chunk_end}] ({len(frames)} frames)...", end=" ", flush=True)

        frames_arr = np.stack(frames)
        with torch.no_grad():
            depths, _ = model.infer_video_depth(
                frames_arr, fps, input_size=518, device=DEVICE, fp32=False)
        del frames_arr

        offset_in_chunk = chunk_start - read_start

        chunk_saved = 0
        for global_idx in range(chunk_start, chunk_end):
            if global_idx % sample_rate == 0:
                local_idx = offset_in_chunk + (global_idx - chunk_start)
                if local_idx < len(depths):
                    depth = depths[local_idx]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.cpu().numpy()

                    # save metric depth (meters)
                    np.save(os.path.join(depth_dir, f"depth_{global_idx:05d}.npy"), depth)

                    # save visualization (clip to 0-80m for display)
                    depth_vis = np.clip(depth, 0, 80)
                    depth_norm = (depth_vis / 80 * 255).astype(np.uint8)
                    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                    cv2.imwrite(os.path.join(depth_dir, f"depth_{global_idx:05d}.png"), depth_colored)
                    chunk_saved += 1

        saved += chunk_saved
        print(f"saved {chunk_saved} depth maps")

        del frames, depths
        gc.collect()
        torch.cuda.empty_cache()

        chunk_start = chunk_end

    cap.release()
    print(f"  Done: {saved} metric depth maps saved")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene1")
    parser.add_argument("--sample_rate", type=int, default=36)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    CHUNK_SIZE = args.chunk_size

    model = load_model()
    estimate_depth_for_scene(model, args.scene, args.sample_rate)
