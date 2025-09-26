import numpy as np
import os
import argparse
import utils
import setup_camera_configuration
import record_from_webcams_with_quicktime
import synchronize_videos
import pose_estimation
import yaml


def record_and_estimate_pose(camera_names, estimator_model = 'coco_base', detector_model = 'coco_base', configuration_number = None, recording_paths = None, synchronize_video = True, model_yaml = './model_paths.yaml', calibration_settings_yaml = './calibration_settings.yaml', checkerboard_display_parameter_yaml = './checkerboard_display_parameters.yaml', origin_camera_idx = 0, script_path = None, project_dir = '', recording_length_seconds = 10, keep_unsynced_files = False):
    if project_dir:
        os.chdir(project_dir)
    else:
        project_dir = os.getcwd()
    if configuration_number is None:
        configuration_number = setup_camera_configuration.configure_cameras(camera_names, calibration_settings_yaml, origin_camera_idx = origin_camera_idx, checkerboard_display_parameter_yaml = checkerboard_display_parameter_yaml, project_dir = project_dir)
    configuration_dir = f'./configurations/{configuration_number}/'
        
    if recording_paths is None:
        input("Press Enter to begin recording. Rembember to create lound noise for synchonization point.")

        run_folder_name = 'recordings'
        run_folder = os.path.join(configuration_dir, run_folder_name)
        record_ID  = utils.create_new_numbered_folder(run_folder)
        recordings_folder = os.path.join(run_folder, str(record_ID))
        
        recording_paths = record_from_webcams_with_quicktime.record_from_cameras(recordings_folder, camera_names, script_path = script_path, recording_length_seconds = recording_length_seconds)
    else:
        recordings_folder = os.path.dirname(recording_paths[0])
        
    if synchronize_video:
        _, recording_paths = synchronize_videos.synchronize_videos(recording_paths, delete_originals = not keep_unsynced_files)
       
        
      
    kpts_2d, heatmaps, kpts_3d = pose_estimation.estimate_pose_from_video(camera_names, recording_paths, estimator_model, detector_model=detector_model, model_yaml = model_yaml, start_end_frames = [0,-1], confidence = 0, extrinsic_params_dir = os.path.join(configuration_dir, 'extrinsic_camera_parameters'))

    #create log to store all relevant file/camera info in one place
    log_dict = {
    'recording_paths': [str(p) for p in recording_paths],
    'kpts_2d': str(os.path.join(recordings_folder, 'kpts_2d.npy')),
    'heatmaps_2d': str(os.path.join(recordings_folder, 'heatmaps_2d.npy')),
    'kpts_3d': str(os.path.join(recordings_folder, 'kpts_3d.npy')),
    'estimator_model': estimator_model,
    'detector_model': detector_model
}
    
    log_path = os.path.join(recordings_folder, 'recording_log.yaml')
    with open(log_path, 'w') as f:
        yaml.dump(log_dict, f)
    
    #these may be nones if certian calculations are skipped
    if kpts_2d is not None:
        np.save(log_dict['kpts_2d'],kpts_2d)
    if heatmaps is not None:
        np.save(log_dict['heatmaps_2d'],heatmaps)
    if kpts_3d is not None:        
        np.save(log_dict['kpts_3d'],kpts_3d)# np.save(os.path.join(run_folder,record_ID,'kpts_3d.npy'),kpts_3d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_names', nargs='+', required=True, help='List of camera names')
    parser.add_argument('--estimator_model')
    parser.add_argument('--detector_model')
    parser.add_argument('--configuration_number', type=int)
    parser.add_argument('--recording_paths', nargs='*')
    parser.add_argument('--synchronize_video', action='store_true')
    parser.add_argument('--model_yaml')
    parser.add_argument('--calibration_settings_yaml')
    parser.add_argument('--checkerboard_display_parameter_yaml')
    parser.add_argument('--origin_camera_idx', type=int)
    parser.add_argument('--script_path')
    parser.add_argument('--project_dir')
    parser.add_argument('--recording_length_seconds', type=int)
    parser.add_argument('--keep_unsynced_files', action='store_true')

    args = parser.parse_args()

    # Convert argparse.Namespace to dict and remove None values
    arg_dict = {k: v for k, v in vars(args).items() if v is not None}

    record_and_estimate_pose(**arg_dict)

if __name__ == "__main__":
    main()