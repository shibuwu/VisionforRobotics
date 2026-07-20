

## Execute
Make sure `Calibration_Imgs/` folder in the same directory as `Wrapper.py`, then run the below comamnd

python Wrapper.py

## Output of running this code 
Results are saved in the `output/` directory 

- `results.txt` — Final K matrix, distortion coefficients, and reprojection error
- `reprojections/` — Images with detected corners (green) and reprojected corners (red) - this shows the errors
- `undistorted/` — Rectified images after removing lens distortion