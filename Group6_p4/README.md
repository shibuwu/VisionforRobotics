# Project 4 — Deep and Un-Deep Visual-Inertial Odometry

Group 6

## Requirements

Python 3.8+, numpy, scipy, matplotlib, opencv-python.
Phase 2 additionally needs PyTorch and Blender (for rendering).

```bash
pip install numpy scipy matplotlib opencv-python torch torchvision
sudo apt install blender ffmpeg
```

## Phase 1 — S-MSCKF

Code is in `Code/Phase_1/`. Uses the EuRoC MH_01_easy dataset.

Download from [ETH ASL](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) in ASL format. Extract so you have `MH_01_easy/mav0/cam0/`, `cam1/`, `imu0/`, `state_groundtruth_estimate0/data.csv`.

Run the VIO:
```bash
cd Code/Phase_1
python vio.py --path /path/to/MH_01_easy --view
```

Evaluate:
```bash
python evaluate.py --est vio_output/trajectory.tum \
    --gt /path/to/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv
```

Make the output video:
```bash
ffmpeg -framerate 30 -i vio_output/frames/frame_%06d.png -c:v libx264 -pix_fmt yuv420p Output.mp4
```

## Phase 2 — Deep VIO

Code is in `Code/Phase_2/`. Split into data generation and training/evaluation.

### Data generation

We generate synthetic training data: a camera looking down at a textured plane with realistic IMU readings. Trajectories are random cubic-spline paths, IMU is simulated at 1000 Hz, and camera frames are rendered in Blender at 100 Hz.

**Step 1 — download textures** (only needed once):
```bash
cd Code/Phase_2
python download_textures.py
```
This creates a `textures/` folder with procedural and OSM map textures used as floor planes.

**Step 2 — generate sequences.** For each sequence, first generate the trajectory and IMU data, then render images in Blender:
```bash
python generate_sequence.py --seed 0 --out_dir dataset_100hz/train/seq_000 --cam_freq 100
blender --background --python render_sequence.py -- --seq_dir dataset_100hz/train/seq_000 --resolution 320 240
```
Repeat with different seeds for all train/test sequences (we used 60 train, 10 test). Test seeds start at 1000 to avoid overlap.

Each sequence folder ends up with:
```
seq_000/
  config.json          # metadata (seed, freq, n_frames, etc.)
  groundtruth.npz      # full 1000 Hz trajectory
  imu.npz              # 1000 Hz gyro + accelerometer
  camera.npz           # 100 Hz camera poses and relative poses
  imu_windows.npz      # pre-sliced IMU between consecutive camera frames
  images/              # rendered RGB frames
  texture_used.png     # the floor texture for this sequence
```

### Training

Compute image normalization stats first:
```bash
python compute_stats_v3.py
```

Train the three networks (vision-only, inertial-only, fusion):
```bash
python train_v2.py --config configs.train_vision_v3
python train_v2.py --config configs.train_inertial_v3
python train_v2.py --config configs.train_fusion_v3
```

Checkpoints are saved to `runs/<experiment_name>/`.

### Evaluation

Run dead-reckoning evaluation on test sequences:
```bash
python run_all_v3.py
```
Results and trajectory plots go to `stage3_results_v3/`.