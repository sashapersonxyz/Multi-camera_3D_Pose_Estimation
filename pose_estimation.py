import os
import numpy as np
import utils
import yaml
import cv2 as cv
from tqdm import tqdm
#ensure that mmpose is in system path sys.path.append('path/to/mmpose-main/demo')
from mmpose_pose_estimation import PoseEstimator
import pickle as pk

def get_pose_3D(camera_params, all_kpts_2d, world_trans_rot = None, camera_indices = None, ignore_nonlinear_distortions = False):
        
    camera_params = camera_params.copy()
    #rearrange camera params to be in order (cmtx,dist,rvec,tvec) instead of (cmtx,rvec,tvec,dist)
    for camera_index in camera_params:
        camera_params[camera_index] = [camera_params[camera_index][0], camera_params[camera_index][3], camera_params[camera_index][1], camera_params[camera_index][2]]
        if ignore_nonlinear_distortions:
            camera_params[camera_index][1] = camera_params[camera_index][1]*0
        
    
    pose_keypoints = range(all_kpts_2d[0].shape[0])
    if camera_indices is None:
        camera_indices = list(camera_params.keys())
    index_positions = [list(camera_params.keys()).index(camera_index) for camera_index in camera_indices]
    #calculate 3d position
    frames_p3ds = []
    for kpts_2d in all_kpts_2d:
        frame_p3ds = []

        for i in range(kpts_2d.shape[0]):
            # Get the 2D slice (indexing with camera_indices transposes the slice for some reason)
            slice_2d = kpts_2d[i, :, index_positions].T
        
            # Find the indices of the top two confidence values for each row
            if slice_2d.shape[0]==3:
                top_indices = np.argsort(slice_2d[2, :])[-2:]
#                print(f'using cameras {[camera_indices[ti] for ti in top_indices]}')
            else:
                top_indices = np.array([0,1])
            # Extract the corresponding points from the first two columns
            top_points = slice_2d[:2,top_indices].T
    
    
            params0 = camera_params[top_indices[0]]
            params1 = camera_params[top_indices[1]]
    
            if any([point is np.nan for point in top_points]):
                p3d = [np.nan, np.nan, np.nan]
            else:
                #p3d = DLT(P0, P1, *list(top_points)) #calculate 3d position of keypoint

                p3d = utils.triangulate_points(top_points,*(params0+params1))
            frame_p3ds.append(p3d)

    
        frame_p3ds = np.array(frame_p3ds).reshape((len(pose_keypoints), 3))
        frames_p3ds.append(frame_p3ds)

    frames_p3ds = np.array(frames_p3ds)
    if world_trans_rot is not None:
        R_W0, T_W0 = world_trans_rot
        frames_p3ds = np.einsum('ij,tpj->tpi', np.linalg.inv(R_W0), frames_p3ds)# + T_W0.reshape(1, 1, 3)


    return frames_p3ds





def get_pose_2D(frames, model, confidence=0.5, pose_keypoints = range(17)):


    #crop to 720x720.
    #Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
    # if frame0.shape[1] != 720:
    #     frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
    #     frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]


    # height = frames[0].shape[0]

    # # To improve performance, optionally mark the image as not writeable to
    # # pass by reference.
    # for frame in frames:
    #     frame.flags.writeable = False

    results = [model(frame) for frame in frames]#[model(frame[:,:,::-1]) for frame in [frame0,frame1]]
    #results = [change_origin(result, height) for result in results]
    
    # #reverse changes
    # for frame in frames:
    #     frame.flags.writeable = True




    #produce standardized format for use with onepose or mmpose
    if model.__module__.startswith('onepose'):
        all_points = [result['points'] for result in results]
        all_confidnces = [result['confidence'].squeeze() for result in results]
    #otherwise assume mmpose
    else:
        all_points = [result[0]['keypoints'].squeeze() for result in results]
        all_confidnces = [result[0]['keypoint_scores'].squeeze() for result in results]



    try:
        heatmaps = [result[1] for result in results]
    except:
        heatmaps = []




    all_frame_keypoints = []
    for points, confidences, frame in zip(all_points, all_confidnces, frames):
        frame_keypoints=[]
        for i, (pxl_x,pxl_y) in enumerate(points):
            if i in pose_keypoints and confidences[i]>=confidence: #only save keypoints that are indicated in pose_keypoints
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                #make sure to flip back in order to display these points correctly
                cv.circle(frame,(pxl_x, pxl_y), 3, (0,0,255), -1)# cv.circle(frame,(pxl_x, height-pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
                kpts = [pxl_x, pxl_y, confidences[i]]
                frame_keypoints.append(kpts)
            else:
                frame_keypoints.append([np.nan, np.nan, confidences[i]])

        all_frame_keypoints.append(frame_keypoints)

    
    # Stack all arrays in 3D for convience
    results_stacked = np.stack([np.concatenate((points, np.expand_dims(confidences,1)), axis=1) for points, confidences in zip(all_points, all_confidnces)], axis=2)



    # uncomment these if you want to see the full keypoints detections
    # mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #
    # mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    for i,frame in enumerate(frames):
        cv.imshow(f'cam{i}', frame)
        cv.resizeWindow(f'cam{i}', frame.shape[1] // 2, frame.shape[0] // 2)

    cv.waitKey(1)

    return results_stacked, heatmaps





def run_pose_est(model, confidence = 0.5, camera_indices = None, recording_paths = None, start_end_frames=[0,-1], frame_shape = [1080, 1920]):
    #containers for detected keypoints for each camera. These are filled at each frame.
    #Streaming: this will run you into memory issue if you run the program without stop
    
    assert (camera_indices is not None) or (recording_paths is not None)
    
    if start_end_frames is None:
        start_end_frames = [0,-1]
    

    heatmaps = []
    
    
    if type(recording_paths) == str:
        recording_paths = dict(zip(camera_indices, [os.path.join(recording_paths,f'camera{ci}') for ci in camera_indices]))

    #can show progress bar if we have defined start/end frames
    total_frames = start_end_frames[1] - start_end_frames[0]
    display_progress_bar = total_frames>0
    if display_progress_bar:
        progress_bar = tqdm(total=total_frames, desc="Processing frame")

    
    kpts_2d = []
    if recording_paths is not None:
        #check if frames were saved as image files
        all_frames = utils.load_frames(recording_paths, start_end_frames)
        for frames in all_frames:
            #frames = [F[frame_num] for F in all_frames.values()]
            frame_2d_kpts, heatmap = get_pose_2D(frames, model, confidence)
            kpts_2d.append(frame_2d_kpts)
            

            heatmaps.append(heatmap)

            # Update progress bar
            if display_progress_bar:
                progress_bar.update(1)


    else:
        #input video streams
        caps = []
        for input_stream in camera_indices:
            caps.append(cv.VideoCapture(input_stream))
        
        for cap in caps:
            cap.set(3, frame_shape[1])
            cap.set(4, frame_shape[0])
    
        while True:
    
            #read frames from stream
            rets, frames = [], []
            for cap in caps:
                ret, frame = cap.read()
                rets.append(ret)
                frames.append(frame)

    
            if any([not ret for ret in rets]): break
    
    
            k = cv.waitKey(1)
            if k & 0xFF == 27: break #27 is ESC key.
            
            frame_2d_kpts = get_pose_2D(frames, model, confidence)
            
            kpts_2d.append(frame_2d_kpts)

        
        for cap in caps:
            cap.release()
    
    # Close progress bar
    if display_progress_bar:
        progress_bar.close()

    
    # Close all OpenCV windows
    utils.destroy_windows_mac()
    
    # Wait for a short period to allow windows to close
    cv.waitKey(1)

    heatmaps = np.array(heatmaps)

    return kpts_2d, heatmaps


def get_yes_or_no_input(prompt):
    """Prompts the user for a 'y' or 'n' response until valid input is received."""
    while True:
        answer = input(prompt).lower().strip()
        if answer in ('y', 'yes'):
            return True
        elif answer in ('n', 'no'):
            return False
        else:
            print("Invalid input. Please answer with 'y' or 'n'.")


def estimate_pose_from_video(camera_names, recording_paths, model, detector_model = 'coco_base', model_yaml = '', start_end_frames = [0,-1], confidence = 0, extrinsic_params_dir = ''):
    #initialize outputs as none for convience
    kpts_2d, heatmaps, kpts_3d = None, None, None
    
    #camera_indicies = list(range(len(recording_paths)))    
    camera_indicies = []
    with open(os.path.join(extrinsic_params_dir, 'camera_names.pkl'), "rb") as f:
        index_name_dict, origin_namera = pk.load(f)

    #invert dict
    index_name_dict = dict([(v,k) for k,v in index_name_dict.items()])
    camera_indicies = [index_name_dict[camera_name] for camera_name in camera_names]
        
    
    
    
    #get all camera parameters
    Ps = {i:None for i in camera_indicies}
    camera_params = {i:None for i in camera_indicies}
        
    for i, camera_name in enumerate(camera_names):
        Ps[i], camera_params[i] = utils.get_params_from_name(camera_name, extrinsic_params_dir = extrinsic_params_dir)
    
    
    
    
    recordings_folder = os.path.dirname(recording_paths[0])
    possible_2dkpts_file = os.path.join(recordings_folder,'kpts_2d.npy')
    if os.path.exists(possible_2dkpts_file):
        answer = get_yes_or_no_input(f"2d keypoints already exist at {possible_2dkpts_file}, do you want to recompute and overwrite (y,n)?")
    if answer:
        if type(model) == str:
            with open(model_yaml, "r") as f:
                model_paths = yaml.safe_load(f)
            
            detector_model_cfg, detector_model_ckpt = model_paths['detectors'][detector_model]
            pose_model_cfg, pose_model_ckpt = model_paths['pose_estimators'][model]
            model_structure = PoseEstimator(detector_model_cfg, detector_model_ckpt, pose_model_cfg, pose_model_ckpt)
            model = model_structure.predict
            
        

        
        
        
        
        
        
    
        recording_paths = {i:recording_paths[i] for i in camera_indicies}
        
        
        
        #first get 2d pose for all cameras for convience
        kpts_2d, heatmaps = run_pose_est(model, confidence = confidence, camera_indices = camera_indicies, recording_paths=recording_paths, start_end_frames=start_end_frames)
        
            
    else:
        kpts_2d = np.load(possible_2dkpts_file)
    
    camera_indices=[0,1]#[0,1]#list(Ps.keys())
    world_trans_rot = None
    #kpts_3d = get_pose_3D(Ps, kpts_2d, pose_keypoints = range(17), camera_indices = camera_indices, world_trans_rot = world_trans_rot)
    kpts_3d = get_pose_3D(camera_params, kpts_2d, camera_indices = camera_indices, world_trans_rot = world_trans_rot)
    
    
    
    
    return kpts_2d, heatmaps, kpts_3d
    
    