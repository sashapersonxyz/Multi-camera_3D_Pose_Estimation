This repo is for end to end 3D human pose estimation from multiple video streams. Here, this task in broken into the following subtasks

1. Computing the intrinsic and extrinsic camera parameters.
2. Capturing video footage of perfromance from each camera and synchronizing video footage accross cameras.
3. Estimating the 2D poses from each camera.
4. Leveraging the camera parameters to determine the 3D pose coordinates.
5. Optionally refinining the estimated pose using the 2D coordinate estimates and optionally smtoothness constraints and ground truth human body measurments. (Additionally optimizing the camera parameters)




Setup:
- Clone the repo: git clone https://github.com/sashapersonxyz/Multi-camera_3D_Pose_Estimation.git
- Install [Mmpose](https://github.com/open-mmlab/mmpose)
- Install additional requirements: pip install requirements.txt
- Modify model_paths.yaml to point to the relevant Mmpose model paths. Note, this code has only been tested with coco type estimation schemes, however others should readily work (3D plotting may not work until the BODYPARTS dictionary is amendend within utils.py)




Camera parameter estimation tips:

- Use a monitor to display the checkerboard if possible
- ensure the checkerboard has a white boarder
- make sure the individual cameras are properly calibrated before attempting stereo calibration
- rotate the board on steep angles to (low RMSE does not imply correct estimation!)
