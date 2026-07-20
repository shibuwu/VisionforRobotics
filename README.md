# Vision for Robotics

Coursework from RBE/CS 549 (Computer Vision) at WPI, Spring 2026. Group 6.

## Projects

### `Group6_p1/` — MyAutoPano: Panorama Stitching
Classical panorama stitching pipeline: Harris corner detection, Adaptive
Non-Maximal Suppression (ANMS), patch feature descriptors, SSD + Lowe ratio
matching, RANSAC homography estimation written from scratch, and
distance-transform blending. Includes stitched outputs for the course test
sets and custom image sets. A deep-learning counterpart (supervised
HomographyNet with 4-point parameterization, trained on synthetic MSCOCO
patch pairs, plus TensorDLT + spatial transformer for the unsupervised
variant) was developed in the same project.

### `Group6_p2/`, `Group6_p2-2/` — Structure from Motion & NeRF ("Buildings Built in Minutes")
- **Phase 1 — SfM from scratch:** RANSAC feature matching, 8-point fundamental
  matrix estimation, essential matrix decomposition, cheirality-based pose
  disambiguation, linear/nonlinear triangulation, PnP with RANSAC, and sparse
  bundle adjustment.
- **Phase 2 — NeRF:** vanilla NeRF in PyTorch with positional encoding and
  hierarchical sampling, trained on the Lego and Ship synthetic scenes at
  400×400 (160k iterations on WPI's Turing cluster).
  Test metrics over 200 views: **Lego 30.53 dB PSNR / 0.958 SSIM**,
  **Ship 29.16 dB PSNR / 0.872 SSIM**. Extra credit: reconstructed a custom
  real-world scene from 52 phone images with COLMAP-estimated poses.

### `Group6_p3/` — EinsteinVision: Monocular AV Perception Pipeline
Tesla-style 3D scene reconstruction from multi-camera driving video:
CLRerNet lane detection fine-tuned on BDD100K, YOLO11 + ByteTrack 2D
tracking, FCOS3D monocular 3D detection, Metric Video-Depth-Anything depth,
RTMPose3D pedestrian pose, Detic open-vocabulary detection, plus speed-limit
sign reading, ground-arrow classification via BEV homography, traffic-light
arrow detection, and brake-light analysis. Scenes rendered in Blender on
WPI's Turing cluster via SLURM. See `Group6_p3/Report.pdf`.

### `Group6_p4/` — Deep Visual-Inertial Odometry
- **Phase 1:** classical stereo MSCKF visual-inertial odometry.
- **Phase 2:** learned odometry on a custom Blender UAV dataset (70
  trajectories, 100 Hz camera / 1000 Hz IMU, 35 textures): pretrained
  ResNet18 vision encoder, 1D-conv inertial encoder over IMU windows with
  Gauss-Markov bias drift, and a late-fusion visual-inertial architecture.
  Achieved 1.82 m vision-only ATE and 1.89 m inertial ATE on held-out
  synthetic trajectories. See `Group6_p4/Report.pdf` and
  `Group6_p4/Group6_DeepVIO.pdf`.

### `hw1_autocalib/` — AutoCalib: Zhang's Camera Calibration
Zhang's calibration implemented from scratch: homography-based intrinsics
initialization, hand-written Rodrigues conversions, sub-pixel corner
refinement, and joint nonlinear refinement of intrinsics + distortion via
least squares. Outputs reprojection overlays and undistorted images.

### `hw0_pblite_cifar10/` — Pb-Lite Boundary Detection & CIFAR-10 Study
Probability-of-boundary edge detection (DoG / Leung-Malik / Gabor filter
banks, texton/brightness/color maps, chi-squared gradients fused with Sobel
and Canny) and a five-architecture CIFAR-10 comparison (custom CNNs,
ResNet-34, ResNeXt-101, DenseNet-121 — best: DenseNet-121 at 93.82% test
accuracy).

## Notes
- Large videos (pitch/demo renders) are excluded from the repo for size.
- Course page: https://rbe549.github.io/spring2026/
