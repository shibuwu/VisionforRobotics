# Phase 1 — How to Run

## Expected Data Layout
Place image folders under:
```
Phase1/
  Data/
    Train/
      Set1/
        1.jpg
        2.jpg
        ...
    Test/
      TestSet1/
        1.jpg
        2.jpg
        ...
```
Images must be `.jpg`.

## Run
From the repo root:
```bash
python3 Phase1/Code/Wrapper.py
```

## Outputs
Results are saved to:
```
Phase1/Phase1_Outputs/<SetName>/
```

Each set folder includes:
- `*_corner.png` (Harris corners)
- `*_anms.png` (ANMS-selected corners)
- `*_FD.png` (feature descriptor visualizations)
- `Matching_*.png` (pre-RANSAC matches)
- `RANSAC_*.png` (post-RANSAC inliers)
- `Panorama_<SetName>.png` (final stitched panorama)

## Notes
- The script processes both `Train` and `Test` folders if they exist.
- If the output canvas grows too large, it is automatically downscaled.
