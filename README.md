This repo is for end to end 3D human pose estimation from multiple video streams. Here, this task in broken into the following subtasks

1. Computing the intrinsic and extrinsic camera parameters.
2. Capturing video footage of perfromance from each camera and synchronizing video footage accross cameras.
3. Estimating the 2D poses from each camera.
4. Leveraging the camera parameters to determine the 3D pose coordinates.
5. Optionally refinining the estimated pose using the 2D coordinate estimates and optionally smtoothness constraints and ground truth human body measurments. (Additionally optimizing the camera parameters)


Setup:
- Clone the repo: git clone https://github.com/sashapersonxyz/Multi-camera_3D_Pose_Estimation.git
- Install [Mmpose](https://github.com/open-mmlab/mmpose) (be sure to add to system path)
- Install additional requirements: pip install requirements.txt
- Modify model_paths.yaml to point to the relevant Mmpose model paths. Note, this code has only been tested with coco type estimation schemes, however others should readily work (3D plotting may not work until the BODYPARTS dictionary is amendend within utils.py)

Calibrating the intrinsic and extrinsic parameters is straightforwardly done when initially running the primary code. In order to ensure accurate calibration, a calibration_settings.yaml needs to be created with the relevant camera and calibration board features. In order to construct the checkerboard calibration imgage, create a checkerboard_display_parameters.yaml based off of the parameters of the display device (monitor, paper, etc). See the respecive example yaml files for reference.

In order to appropriately reference the attached cameras for recording in QuickTime, list  available recording device names by checking QuckTime or running the following ffmpeg command: ffmpeg _-f avfoundation -list_devices true -i ""_. These correspond to "camera_name0" and "camera_name1" below

    _python record_and_estimate_pose.py --camera_names "camera_name0" "camera_name1" --recording_length_seconds 20 --synchronize_video_

After following all user prompts, the result of this code will be intrinsic and extrinsic camera calibration files, synchroized video recordings, and 2D/3D/heatmap pose estimation for the captured performance.

The raw estimated pose can be fairly jittery. In order to refine the estimated pose we can run the command

    _python pose_refinement.py --run_path path/to/recording_1 path/to/recording_2 --refinement_types SGD linear_interpolation_

to perfrom SGD (refinement_types = SGD) to produce MLE estimates under constraints (body length specifications or smoothness assumptions) or linear interpolation with outlier filtering (refinement_types = linear_interpolation). See pose_refinement.linear_interpolation and pose_refinement.Optimized_3d_Pose_Estimation for more impolementation details. Specific refinement parameters can be specified by creating a parameters yaml file containing parameters for SGD and/or linear_interpolation. e.g.
SGD:
  max_iter: 50000
  lr: 0.01
  lambda_smooth: 0.000001
  lambda_body_length: 1
  patience: 100
  time_interval: [0,400]

When perfoming SGD with body length constraints, any relevant body length constraints can be passed by passing a yaml file under --body_part_lengths_yaml of the form 
individual_specifier:
  body_segment_name_0: body_segment_0_length_mm
  .
  .
  body_segment_name_n: body_segment_n_length_mm

heatmaps, 3D projective plots, and  be constructed using

    _python plot_utils.py --recording_log path/to/recording_log.yaml --plot_types "3D_pose"_.

and 3D interactive plots can be viewed using plot_utils.interactive_3d_pose_animation.

See the help messages for record_and_estimate_pose.py, pose_refinement.py, and plot_utils.py for more information.


Camera parameter estimation tips:

- Use a monitor to display the checkerboard if possible
- ensure the checkerboard has a white boarder
- make sure the individual cameras are properly calibrated before attempting stereo calibration
- rotate the board on steep angles to (low RMSE does not imply correct estimation!)
