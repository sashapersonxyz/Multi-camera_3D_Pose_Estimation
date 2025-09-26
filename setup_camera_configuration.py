##if on mac, be sure to give camera permissions before running otherwise opencv will crash this.. https://stackoverflow.com/questions/72756091/opencv-not-authorized-to-capture-video-mac


import cv2 as cv
import numpy as np
import os
import utils

import matplotlib.pyplot as plt
import time

import pickle as pk
from datetime import datetime
import yaml

#import imageio














#possible camera names (used as unique camera ID since cv2 can't do this for some reason)
#possible_names = ["pixel_ultrawide", "pixel_wide", "nexigo", "microsoft", "emeet", "usb"]

def select_webcam_names(possible_names, save_dir = ''):
    if not save_dir:
        save_dir = os.getcwd()
    pickle_file = os.path.join(save_dir,'extrinsic_camera_parameters/camera_names.pkl')
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            cameras, origin_camera = pk.load(f)
    else:        
        cameras = {}
        for i in range(10):  # Assuming at most 10 cameras
            cap = cv.VideoCapture(i)
            if not cap.isOpened():
                break
            print("Available cameras:")
            for idx, name in enumerate(possible_names, start=1):
                print(f" {idx}. {name}")
    
            # Get the camera resolution
            width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera {i} resolution: {int(width)}x{int(height)}")
    
            print(f"Camera {i}:")
            
            # Display the camera feed
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv.imshow(f"Camera {i}", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
    
            # Prompt for camera name
            while True:
                try:
                    selected_idx = int(input(f"Enter the number of the camera (1-{len(possible_names)}) or 0 to skip: "))
                    if selected_idx == 0:
                        break
                    selected_name = possible_names[selected_idx - 1]
                    cameras[i] = selected_name
                    break
                except (ValueError, IndexError):
                    print("Invalid input. Please enter a valid number.")
    
            utils.destroy_windows_mac()
            cap.release()
        
        
        while True:
            try:
                selected_idx = int(input(f"Enter the number of any additional (unconnected) cameras (1-{len(possible_names)}) or 0 to skip: "))
                if selected_idx == 0:
                    break
                selected_name = possible_names[selected_idx - 1]
                cameras[i] = selected_name
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid number.")

    
        while True:
            try:
                origin_camera = int(input(f"Enter the number of the origin camera among {cameras}: "))
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter a valid number.")
        if not os.path.exists(os.path.join(config_dir,'extrinsic_camera_parameters')):
            os.makedirs(os.path.join(config_dir,'extrinsic_camera_parameters'))
        with open(pickle_file, "wb") as f:
            pk.dump((cameras, origin_camera), f)

    return cameras, origin_camera



def display_camera(k):
    cap = cv.VideoCapture(k)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cv.startWindowThread()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow(f"Camera {k}", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    utils.destroy_windows_mac()


def change_origin(result, height):
    arr = result['points']
    arr[:, 1] = height - arr[:, 1]
    return {'points': arr, 'confidence': result['confidence']}






#old fcn didnt use the distortion parameters
# def get_pose_3D(Ps, all_kpts_2d, pose_keypoints = range(17), world_trans_rot = None, camera_indices = None):

    
#     if camera_indices is None:
#         camera_indices = list(range(all_kpts_2d[0].shape[2]))

#     #calculate 3d position
#     frames_p3ds = []
#     for kpts_2d in all_kpts_2d:
#         frame_p3ds = []

#         for i in range(kpts_2d.shape[0]):
#             # Get the 2D slice (indexing with camera_indices transposes the slice for some reason)
#             slice_2d = kpts_2d[i, :, camera_indices].T
        
#             # Find the indices of the top two confidence values for each row
#             top_indices = np.argsort(slice_2d[2, :])[-2:]
#             print(f'using cameras {[camera_indices[ti] for ti in top_indices]}')
            
#             # Extract the corresponding points from the first two columns
#             top_points = slice_2d[:2,top_indices].T
    
    
#             P0 = Ps[top_indices[0]]
#             P1 = Ps[top_indices[1]]
    
#             if any([point is np.nan for point in top_points]):
#                 p3d = [np.nan, np.nan, np.nan]
#             else:
#                 p3d = DLT(P0, P1, *list(top_points)) #calculate 3d position of keypoint
#             frame_p3ds.append(p3d)

    
#         frame_p3ds = np.array(frame_p3ds).reshape((len(pose_keypoints), 3))
#         frames_p3ds.append(frame_p3ds)

#     frames_p3ds = np.array(frames_p3ds)
#     if world_trans_rot is not None:
#         R_W0, T_W0 = world_trans_rot
#         frames_p3ds = np.einsum('ij,tpj->tpi', np.linalg.inv(R_W0), frames_p3ds)# + T_W0.reshape(1, 1, 3)


#     return frames_p3ds





plt.style.use('seaborn-v0_8')



def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts



from PIL import Image




def create_black_white_grid(k, r, c, overall_height, overall_width):

    # Create a new image with mode 'L' (greyscale) and white background
    new_img = Image.new('L', (overall_width, overall_height), color=255)

    # Create a new image with mode '1' (1-bit pixels, black and white)
    img = Image.new('1', (c*k, r*k))

    # Create a pixel access object to manipulate pixels
    pixels = img.load()

    # Iterate over each square
    for i in range(r):
        for j in range(c):
            # Determine the color of the square based on its position
            color = 0 if (i + j) % 2 == 0 else 255

            # Set the color of each pixel in the square
            for x in range(k):
                for y in range(k):
                    pixels[j*k + x, i*k + y] = color

    # Calculate the position to paste the pattern
    left = (overall_width - img.width) // 2
    top = (overall_height - img.height) // 2

    # Paste the pattern onto the new image
    new_img.paste(img, (left, top))

    return new_img








def configure_cameras(camera_names, calibration_settings_yaml, project_dir, origin_camera_idx = 0, checkerboard_display_parameter_yaml = ''):

    
    configuration_number = utils.create_new_numbered_folder(os.path.join(project_dir, 'configurations'))
    global config_dir
    config_dir = os.path.join(project_dir, 'configurations', str(configuration_number))

    
        
    # #numbers for ipad
    # r = 5   # Number of rows
    # c = 8   # Number of columns
    # boarder = 20 #size of grew boarder in pixels
    # height = 1668  # Overall height of the final image
    # width = 2224   # Overall width of the final image
    # ppmm = 10.393701 # pixels per mm
    
    
    # #numbers for tv
    # r = 5   # Number of rows
    # c = 8   # Number of columns
    # boarder = 20 #size of grew boarder in pixels
    # height = 2160  # Overall height of the final image
    # width = 3840   # Overall width of the final image
    # width_mm = 941.184
    # ppmm = width/width_mm
    with open(checkerboard_display_parameter_yaml, "r") as f:
        params = yaml.safe_load(f)
    
    #calculate the width in pixels of each square (note, the min is taken so that all the squares fit!)
    k = int(np.floor(min([params['width']/params['c'], params['height']/params['r']]))-params['boarder'])
    
    if not os.path.exists(os.path.join(project_dir,'checkerboard_pattern.jpg')):
        

        img = create_black_white_grid(k, params['r'], params['c'], params['height'], params['width'])
        img.save('./checkerboard_pattern.jpg')
    
    
    
    
    
    
    
    
    
    #This will contain the calibration settings from the calibration_settings.yaml file
    with open(calibration_settings_yaml, "r") as f:
        calibration_settings = yaml.safe_load(f)
    
    #get camera names 
    selected_cameras, origin_camera = select_webcam_names(camera_names, save_dir = config_dir)
    origin_camera_name = selected_cameras[origin_camera]
    camera_names = list(selected_cameras.values())
    origin_camera_idx = camera_names.index(origin_camera_name)
    
    
    # calibration_settings = {
    #     'frame_width': 1920,
    #     'frame_height': 1080,
    #     'mono_calibration_frames': 20,
    #     'stereo_calibration_frames': 10,
    #     'view_resize': 2,
    #     'checkerboard_box_size_scale': None,
    #     'checkerboard_rows': 4,
    #     'checkerboard_columns': 7,
    #     'cooldown': 100
    # }
    if not 'checkerboard_box_size_scale' in calibration_settings:
        #we need to know either th pixel density or the overall width of the display in order to calculate the checkboard box size    
        assert 'ppmm' in params or 'width_mm' in params
        
        if not 'ppmm' in params:
            params['ppmm'] = params['width']/params['width_mm']
            
        #convert display screen value to cm
        calibration_settings['checkerboard_box_size_scale']=k/(10*params['ppmm'])
        
    
    calibration_settings = {**calibration_settings,**{v: k for k, v in selected_cameras.items()}}
    
    utils.set_calibration_settings(calibration_settings)
    
    
    cmtx_dist = []

    for camera_name in camera_names:
        if not(os.path.exists(os.path.join(project_dir, 'intrinsic_camera_parameters',camera_name+'.dat'))):
            print(f'calibrating camera: {camera_name}')
            """Step1. Save calibration frames for single cameras"""
            utils.save_frames_single_camera(camera_name, project_dir) #save frames for camera_name
    
            
            """Step2. Obtain camera intrinsic matrices and save them"""
            #camera_name intrinsics
            images_prefix = os.path.join(project_dir,'frames', f'{camera_name}*')
            cmtx, dist = utils.calibrate_camera_for_intrinsic_parameters(images_prefix) 
            utils.save_camera_intrinsics(cmtx, dist, camera_name, root_path=project_dir) #this will write cmtx and dist to disk
            cmtx_dist.append([cmtx, dist])
        else:
            cmtx_dist.append(list(utils.read_camera_parameters(camera_name, params_dir = os.path.join(project_dir, 'intrinsic_camera_parameters'))))
    
    os.chdir(config_dir)
    for idx, camera_name in enumerate(camera_names):
        origin_camera_saved = os.path.exists(os.path.join(config_dir,f'extrinsic_camera_parameters/rot_trans_{origin_camera_name}'+ '.dat'))
        if idx != origin_camera_idx:
            if not os.path.exists(os.path.join(config_dir,f'extrinsic_camera_parameters/rot_trans_{camera_name}'+ '.dat')):
                
                cmtx0, dist0, = cmtx_dist[origin_camera_idx]
                cmtx1, dist1 = cmtx_dist[idx]
                
                
                response = input(f"Do you want to provide translation and rotation values for camera {camera_name} (alternatively these will be computed if the camera is accessable to python)? (y/n): ").strip().lower()
                
                if response == 'y':
                    # Ask for the translation values
                    X = float(input("Enter the X translation (mm): "))
                    Y = float(input("Enter the Y translation (mm): "))
                    Z = float(input("Enter the Z translation (mm): "))
                    
                    # Ask for the right triangle lengths (for rotation)
                    Z_triangle = float(input("Measuring the angle of the camera, asuming lying in the X-Z plane, enter the length of the adjacent side (Z)) (mm): "))
                    X_triangle = float(input("Enter the length of the opposite side (X) (mm): "))
                    T, R = utils.compute_extrinsic_from_measurments([X,Y,Z], X_triangle, Z_triangle)
                else:
                    
                    """Step3. Save calibration frames for both cameras simultaneously"""
                    utils.save_frames_two_cams(origin_camera_name, camera_name) #save simultaneous frames
                    
                    
                    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
                    frames_prefix_c0 = os.path.join('frames_pair', f'{origin_camera_name}*')
                    frames_prefix_c1 = os.path.join('frames_pair', f'{camera_name}*')
                    R, T = utils.stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)
                    
                    
                """Step5. Set origin calibration data where camera0 defines the world space origin."""
                #camera0 rotation and translation is identity matrix and zeros vector
                R0 = np.eye(3, dtype=np.float32)
                T0 = np.array([0., 0., 0.]).reshape((3, 1))
                
                utils.save_extrinsic_calibration_parameters(R, T, camera_name, root_dir=config_dir) #this will write R and T to disk
                if origin_camera_saved == False:
                    utils.save_extrinsic_calibration_parameters(R0, T0, origin_camera_name, root_dir=config_dir) #this will write R and T to disk
                    origin_camera_saved = True
                R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
                #check your calibration makes sense
                camera0_data = [cmtx0, dist0, R0, T0]
                camera1_data = [cmtx1, dist1, R1, T1]
                
                try:
                    _zshift = 400.
                    print(f'Displaying marker at (0,0,{_zshift}cm) from origin. Press Esc to end.')
                    utils.check_calibration(origin_camera_name, camera0_data, camera_name, camera1_data, _zshift = _zshift)
                except:
                    pass
                

    return configuration_number    


