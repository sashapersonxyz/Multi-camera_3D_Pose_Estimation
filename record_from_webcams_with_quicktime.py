import os
#import to run bash script in order to record webcams through quicktime
import subprocess
import shlex


def run_recording_script(save_paths, camera_names, script_path = None, recording_length_seconds = 10):
    """
    Run the record webcams script.

    Args:
        script_path (str): Path to the bash script.
        save_path1 (str): First output file path (e.g., for webcam 1).
        save_path2 (str): Second output file path (e.g., for webcam 2).
        recording_length_seconds (int): length of video recording in seconds
    """
    try:
        if script_path is None:
            current_dir = os.getcwd()
            script_path = os.path.join(current_dir, 'quicktime_record_streams.sh')
            

        
        cmd = f'bash {shlex.quote(script_path)}'+' '+' '.join([shlex.quote(save_path) for save_path in save_paths])+' '+' '.join([shlex.quote(camera_name) for camera_name in camera_names])+f' {recording_length_seconds}'
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Script exited with code {result.returncode}")
        else:
            print("Script executed successfully.")
    except Exception as e:
        print(f"Error running script: {e}")
        
def record_from_cameras(recordings_folder, camera_names, script_path = None, recording_length_seconds = 10):
    '''camera_names: list of strings of camera names which match the names in quicktime'''
    #rn the script is only setup to handle 2 input streams
    assert len(camera_names) == 2
    
    #construct save paths
    save_paths = [os.path.join(recordings_folder, camera_name+'.mov') for camera_name in camera_names]
    #ensure absolute path instead of relative references
    for i, path in enumerate(save_paths):
        save_paths[i] = os.path.abspath(path)
    run_recording_script(save_paths, camera_names, script_path = None, recording_length_seconds = recording_length_seconds)
    

    return save_paths
