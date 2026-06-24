# Project 2 -- SfM and NeRF

Shibani Senthilbabu and Ajay Adit Jagan
WPI, RBE/CS 549

## Phase 1: Classical Structure from Motion

```bash
cd Phase1
python Wrapper.py
```

Place the P2Data folder (with `calibration.txt`, `matching1-4.txt`, and images `1-5.png`) inside `Phase1/P2Data/`.
Output figures are saved to `Data/IntermediateOutputImages/`.

## Phase 2: NeRF

### Training

```bash
cd Phase2
python Wrapper.py --scene lego
```

### Evaluation

```bash
python Wrapper.py --scene lego --eval --ckpt checkpoints/lego/ckpt_150000.pt
```

### Training without Positional Encoding

```bash
python Wrapper.py --scene lego --no_pe
```

### Generating GIFs

```bash
python gif_generator.py
```


### Data

Place the NeRF synthetic dataset in `Phase2/nerf_synthetic/` (e.g. `Phase2/nerf_synthetic/lego/`).

### Extra Credit: Custom Scene

We collected a custom dataset and used COLMAP to obtain camera poses. The `colmap2nerf.py` script converts COLMAP output to NeRF format.
