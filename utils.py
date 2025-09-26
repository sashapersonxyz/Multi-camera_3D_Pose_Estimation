import numpy as np
import cv2 as cv
import glob
from scipy import linalg
import yaml
import os
import torch
import inspect


#this is needed to destroy windows on mac
def destroy_windows_mac():
    cv.destroyAllWindows()
    for i in range (1,5):
        cv.waitKey(1)


#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):
    try:
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
    
        B = A.transpose() @ A
        U, s, Vh = linalg.svd(B, full_matrices = False)
    except:
        print([point1, point2])
    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename, selected_cameras):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        calibration_settings = None
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)
    
    calibration_settings = {**calibration_settings,**{v: k for k, v in selected_cameras.items()}}
    
def set_calibration_settings(S):
    global calibration_settings
    calibration_settings = S


#Open camera stream and save frames
def save_frames_single_camera(camera_name, root_dir):

    save_dir = os.path.join(root_dir,'frames')
    #create frames directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    #open video stream and change resolution.
    #Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
    
        ret, frame = cap.read()
        if ret == False:
            #if no video data is received, can't calibrate the camera, so exit.
            print("No video data received from camera. Exiting...")
            quit()

        frame_small = cv.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    # Close all OpenCV windows
    cv.destroyAllWindows()
    
    # Wait for a short period to allow windows to close
    cv.waitKey(1)


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

    
    # Close all OpenCV windows
    cv.destroyAllWindows()
    
    # Wait for a short period to allow windows to close
    cv.waitKey(1)

    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, root_path=None):

    intrinsic_parameter_folder = 'intrinsic_camera_parameters'
    
    if root_path is None:
        root_path = os.getcwd()
        
    intrinsic_parameter_path = os.path.join(root_path, intrinsic_parameter_folder) 
    
    #create folder if it does not exist
    if not os.path.exists(intrinsic_parameter_path):
        os.mkdir(intrinsic_parameter_path)
    out_filename = os.path.join(intrinsic_parameter_path, camera_name+ '.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')









def find_checkerboard(frame):
    # Calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    # Find chessboard corners in the enhanced image
    c_ret, corners = cv.findChessboardCorners(gray, (rows, columns),flags=(cv.CALIB_CB_FAST_CHECK +
                                                  cv.CALIB_CB_ADAPTIVE_THRESH +
                                                  cv.CALIB_CB_NORMALIZE_IMAGE))

    return gray, c_ret, corners


#open both cameras and take calibration frames
def save_frames_two_cams(camera0_name, camera1_name):

    #create frames directory
    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    #settings for taking data
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Cameras not returning video data. Exiting...')
            quit()

        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame0)

                savename = os.path.join('frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame1)
    
                #check if we can actually detect the checkerboard                
                gray1, c_ret1, corners1 = find_checkerboard(frame0)
                gray2, c_ret2, corners2 = find_checkerboard(frame1)
                if c_ret1 and c_ret2:
                    saved_count += 1
                else:
                    print(f'CANT DETECT CHECKERBOARD ON CAMERA(S) {np.array([camera0_name, camera1_name])[np.where([not c_ret1, not c_ret2])]}')
                cooldown = cooldown_time
                    
        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

        #break out of the loop when enough number of frames have been saved
        if saved_count == number_to_save: break

    # Close all OpenCV windows
    cv.destroyAllWindows()
    
    # Wait for a short period to allow windows to close
    cv.waitKey(1)



#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 10e-5)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for idx, (frame0, frame1) in enumerate(zip(c0_images, c1_images)):
        
        gray1, c_ret1, corners1 = find_checkerboard(frame0)
        gray2, c_ret2, corners2 = find_checkerboard(frame1)
        
        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
        else:
            print(f'couldnt find the checkerboard in frame {idx}')
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    # Close all OpenCV windows
    cv.destroyAllWindows()
    
    # Wait for a short period to allow windows to close
    cv.waitKey(1)

    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P


def project_points(points_3d, K, R, T, dist_coeffs=None):
    """
    Project 3D points to 2D image points given camera intrinsics and extrinsics.
    points_3d: (N,3)
    R: (3,3)
    T: (3,) or (3,1)
    K: (3,3)
    dist_coeffs: None or (k,)
    """
    points_3d_shape = points_3d.shape
    rvec, _ = cv.Rodrigues(R)
    points_3d = np.asarray(points_3d, dtype=float).reshape(-1, 1, 3)
    T = np.asarray(T, dtype=float).reshape(3, 1)
    pts2d, _ = cv.projectPoints(points_3d, rvec, T, K, dist_coeffs)
    
    if len(points_3d_shape)==3:
        pts2d = pts2d.reshape(points_3d_shape[:2]+(2,))
    else:
        pts2d = pts2d.reshape(-1, 2)
    
    return pts2d




# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27: break
    # Close all OpenCV windows
    cv.destroyAllWindows()
    
    # Wait for a short period to allow windows to close
    cv.waitKey(1)




#projects 3d point to 2d camera coordintes. DOES NOT LEVERAGE DISTORTION!
def compute_2d_coordinates(P, point_3d):

    # Add a z-shift to the point to project it in front of the cameras
    point_3d_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.])

    # Project the 3D point to the 2D image plane
    uv = P @ point_3d_homo
    uv = np.array([uv[0], uv[1]]) / uv[2]

    return uv




def check_calibration_all_cameras(camera_names, _zshift=50.):
    def get_projection_matrix(cmtx, R, T):
        return cmtx @ np.hstack((R, T))

    draw_axes_points = np.array([[0., 0., 0.],
                                  [1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]])
    z_shift = np.array([0., 0., _zshift]).reshape((1, 3))
    draw_axes_points = 5 * draw_axes_points + z_shift

    pixel_points_cameras = {}
    for camera_name in camera_names:
        cmtx, dist = read_camera_parameters(camera_name)
        R, T = read_rotation_translation(camera_name)
        
        pixel_points_camera = []
        for _p in draw_axes_points:
            uv = compute_2d_coordinates(_p)
            pixel_points_camera.append(uv)

        pixel_points_cameras[camera_name] = np.array(pixel_points_camera)

    # Assuming calibration_settings is a dictionary containing camera settings
    frame_width = calibration_settings['frame_width']
    frame_height = calibration_settings['frame_height']

    caps = {}
    for camera_name in camera_names:
        cap = cv.VideoCapture(calibration_settings[camera_name])
        cap.set(3, frame_width)
        cap.set(4, frame_height)
        caps[camera_name] = cap

    while True:
        frames = {}
        for camera_name, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f'Video stream for {camera_name} not returning frame data')
                quit()
            frames[camera_name] = frame

        for camera_name, pixel_points_camera in pixel_points_cameras.items():
            frame = frames[camera_name]
            origin = tuple(pixel_points_camera[0].astype(np.int32))
            for col, _p in zip([(0, 0, 255), (0, 255, 0), (255, 0, 0)], pixel_points_camera[1:]):
                _p = tuple(_p.astype(np.int32))
                cv.line(frame, origin, _p, col, 2)
            cv.imshow(camera_name, frame)

        k = cv.waitKey(1)
        if k == 27:
            break

    cv.destroyAllWindows()
    cv.waitKey(1)










def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1


def compute_extrinsic_from_measurments(XYZ, X_len, Z_len):
    '''compute an approximation of the extrinsic camera parameters from XYZ position and lengths of the XZ right triangle (assuming parallel to Y)'''
    translation_vect = np.array(XYZ)
    translation_vect = translation_vect[:,None]

    hyp = np.sqrt(np.sum([X**2 for X in [X_len, Z_len]]))
    
    Ctheta, Stheta = Z_len/hyp, X_len/hyp
    
    rotation_matrix = np.array([
    [Ctheta, 0, Stheta],
    [0, 1, 0],
    [-Stheta, 0, Ctheta]
])
    return translation_vect, rotation_matrix


def save_extrinsic_calibration_parameters(R, T, camera_name, root_dir = None):
    
    if root_dir is None:
        root_dir = os.getcwd()
    
    extrinsic_parameter_folder = os.path.join(root_dir, 'extrinsic_camera_parameters')
    
    #create folder if it does not exist
    if not os.path.exists(extrinsic_parameter_folder):
        os.mkdir(extrinsic_parameter_folder)

    camera0_rot_trans_filename = os.path.join(extrinsic_parameter_folder,f'rot_trans_{camera_name}.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R, T


def read_camera_parameters(camera_name , params_dir = ''):
    if not params_dir:
        params_dir = os.getcwd()
    
    inf = open(os.path.join(params_dir, camera_name + '.dat'), 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_name, params_dir = ''):
    if not params_dir:
        params_dir = os.getcwd()
    
    inf = open(os.path.join(params_dir,'rot_trans_'+ camera_name + '.dat'), 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def calculate_projection_matrix(cmtx, rvec, tvec):
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def get_params_from_name(camera_name, intrinsic_params_dir = '', extrinsic_params_dir = ''):
    if not intrinsic_params_dir:
        intrinsic_params_dir = os.path.join(os.getcwd(), 'intrinsic_camera_parameters')
    if not extrinsic_params_dir:
        extrinsic_params_dir = os.path.join(os.getcwd(), 'extrinsic_camera_parameters')    
    #read camera parameters
    cmtx, dist, rvec, tvec, P = None, None, None, None, None
    try:        
        cmtx, dist = read_camera_parameters(camera_name, params_dir=intrinsic_params_dir)
    except:
        print(f'failed to load {camera_name} intrinsic params')
    try:        
        rvec, tvec = read_rotation_translation(camera_name, params_dir = extrinsic_params_dir)
    except:
        print(f'failed to load {camera_name} extrinsic params')

    #calculate projection matrix
    try:
        P = calculate_projection_matrix(cmtx, rvec, tvec)
    except:
        print(f'failed to compute {camera_name} projection')
    return P, [cmtx,rvec,tvec,dist]

def write_keypoints_to_disk(filename, kpts):
    directory = os.path.dirname(filename)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()



def frame_generator(recording_paths, start_end_frames=[0, -1]):
    camera_indices = list(recording_paths.keys())
    def process_image_files(camera_index, start, end):
        path = recording_paths[camera_index]
        filenames = sorted(
            [f for f in os.listdir(path) if f.endswith('jpg')],
            key=lambda x: int(x.split("frame")[1].split(".")[0])
        )
        filenames = filenames[start:end]
        for filename in filenames:
            frame = cv.imread(os.path.join(path, filename))
            yield cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    def process_video_files(video_path, start, end):
        for frame in read_video_as_frames(video_path)[start:end]:
            yield cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    def read_video_as_frames(video_path):
        cap = cv.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    

    if os.path.exists(recording_paths[camera_indices[0]]):
        # Create iterators for each camera
        iterators = {}
        for camera_index in camera_indices:
            video_path = recording_paths[camera_index]
            iterators[camera_index] = process_video_files(video_path, start_end_frames[0], start_end_frames[1])

        while True:
            frames_list = []
            finished = True
            for camera_index, it in iterators.items():
                try:
                    frame = next(it)
                    frames_list.append(frame)
                    finished = False
                except StopIteration:
                    frames_list.append(None)
            if finished:
                break
            yield frames_list
    else:
        raise FileNotFoundError("Error loading video")




def load_frames(recording_paths, start_end_frames=[0, -1]):
    if not(type(recording_paths) in {list, dict}):
        return None
    if type(recording_paths) == list:
        recording_paths = dict(enumerate(recording_paths))
    return frame_generator(recording_paths, start_end_frames)




# Copyright (c) OpenMMLab. All rights reserved.
def convert_keypoint_definition(keypoints, pose_det_dataset,
                                pose_lift_dataset):
    """Convert pose det dataset keypoints definition to pose lifter dataset
    keypoints definition, so that they are compatible with the definitions
    required for 3D pose lifting.

    Args:
        keypoints (ndarray[K, 2 or 3]): 2D keypoints to be transformed.
        pose_det_dataset, (str): Name of the dataset for 2D pose detector.
        pose_lift_dataset (str): Name of the dataset for pose lifter model.

    Returns:
        ndarray[K, 2 or 3]: the transformed 2D keypoints.
    """
    assert pose_lift_dataset in [
        'Body3DH36MDataset', 'Body3DMpiInf3dhpDataset'
        ], '`pose_lift_dataset` should be `Body3DH36MDataset` ' \
        f'or `Body3DMpiInf3dhpDataset`, but got {pose_lift_dataset}.'

    coco_style_datasets = [
        'TopDownCocoDataset', 'TopDownPoseTrack18Dataset',
        'TopDownPoseTrack18VideoDataset'
    ]
    keypoints_new = np.zeros((17, keypoints.shape[1]), dtype=keypoints.dtype)
    if pose_lift_dataset == 'Body3DH36MDataset':
        if pose_det_dataset in ['TopDownH36MDataset']:
            keypoints_new = keypoints
        elif pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
            # rearrange other keypoints
            keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
                
            
            #the above permutations are incorrect in my case for some reason
            perm_order = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]
            inverse_perm = [perm_order.index(i) for i in range(len(perm_order))]
            keypoints_new = keypoints_new[[inverse_perm]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[9] + keypoints[6]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[3] + keypoints[0]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[6, 7, 8, 9, 10, 11, 3, 4, 5, 0, 1, 2]]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[0] = (keypoints[6] + keypoints[7]) / 2
            # thorax is in the middle of l_shoulder and r_shoulder
            keypoints_new[8] = (keypoints[0] + keypoints[1]) / 2
            # spine is in the middle of thorax and pelvis
            keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
            # neck base (top end of neck) is 1/4 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[9] = (3 * keypoints[13] + keypoints[12]) / 4
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[10] = (5 * keypoints[13] + 7 * keypoints[12]) / 12

            keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
                keypoints[[7, 9, 11, 6, 8, 10, 0, 2, 4, 1, 3, 5]]
        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    elif pose_lift_dataset == 'Body3DMpiInf3dhpDataset':
        if pose_det_dataset in coco_style_datasets:
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[11] + keypoints[12]) / 2
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[5] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2

            # in COCO, head is in the middle of l_eye and r_eye
            # in PoseTrack18, head is in the middle of head_bottom and head_top
            keypoints_new[16] = (keypoints[1] + keypoints[2]) / 2

            if 'PoseTrack18' in pose_det_dataset:
                keypoints_new[0] = keypoints[1]
                # don't extrapolate the head top confidence score
                keypoints_new[16, 2] = keypoints_new[0, 2]
            else:
                # head top is extrapolated from neck and head
                keypoints_new[0] = (4 * keypoints_new[16] -
                                    keypoints_new[1]) / 3
                # don't extrapolate the head top confidence score
                keypoints_new[0, 2] = keypoints_new[16, 2]
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15
            ]]
        elif pose_det_dataset in ['TopDownAicDataset']:
            # head top is head top
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is neck
            keypoints_new[1] = keypoints[13]
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[9] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[0:12]
        elif pose_det_dataset in ['TopDownCrowdPoseDataset']:
            # head top is top_head
            keypoints_new[0] = keypoints[12]
            # neck (bottom end of neck) is in the middle of
            # l_shoulder and r_shoulder
            keypoints_new[1] = (keypoints[0] + keypoints[1]) / 2
            # pelvis (root) is in the middle of l_hip and r_hip
            keypoints_new[14] = (keypoints[7] + keypoints[6]) / 2
            # spine (centre of torso) is in the middle of neck and root
            keypoints_new[15] = (keypoints_new[1] + keypoints_new[14]) / 2
            # head (spherical centre of head) is 7/12 the way from
            # neck (bottom end of neck) to head top
            keypoints_new[16] = (5 * keypoints[13] + 7 * keypoints[12]) / 12
            # arms and legs
            keypoints_new[2:14] = keypoints[[
                1, 3, 5, 0, 2, 4, 7, 9, 11, 6, 8, 10
            ]]

        else:
            raise NotImplementedError(
                f'unsupported conversion between {pose_lift_dataset} and '
                f'{pose_det_dataset}')

    return keypoints_new



global CONNECTIVITY_DICT
CONNECTIVITY_DICT = {
    'cmu': [(0, 2), (0, 9), (1, 0), (1, 17), (2, 12), (3, 0), (4, 3), (5, 4), (6, 2), (7, 6), (8, 7), (9, 10), (10, 11), (12, 13), (13, 14), (15, 1), (16, 15), (17, 18)],
    'coco': [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (5, 11), (6, 12), (11, 12)],
    "mpii": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (8, 12), (8, 13), (10, 11), (11, 12), (13, 14), (14, 15)],
    "human36m": [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 9), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)],
    "kth": [(0, 1), (1, 2), (5, 4), (4, 3), (6, 7), (7, 8), (11, 10), (10, 9), (2, 3), (3, 9), (2, 8), (9, 12), (8, 12), (12, 13)],
}

global POINT_INFO
POINT_INFO = {'coco':{0: {'name': 'nose',
  'id': 0,
  'color': [51, 153, 255],
  'type': 'upper',
  'swap': ''},
 1: {'name': 'left_eye',
  'id': 1,
  'color': [51, 153, 255],
  'type': 'upper',
  'swap': 'right_eye'},
 2: {'name': 'right_eye',
  'id': 2,
  'color': [51, 153, 255],
  'type': 'upper',
  'swap': 'left_eye'},
 3: {'name': 'left_ear',
  'id': 3,
  'color': [51, 153, 255],
  'type': 'upper',
  'swap': 'right_ear'},
 4: {'name': 'right_ear',
  'id': 4,
  'color': [51, 153, 255],
  'type': 'upper',
  'swap': 'left_ear'},
 5: {'name': 'left_shoulder',
  'id': 5,
  'color': [0, 255, 0],
  'type': 'upper',
  'swap': 'right_shoulder'},
 6: {'name': 'right_shoulder',
  'id': 6,
  'color': [255, 128, 0],
  'type': 'upper',
  'swap': 'left_shoulder'},
 7: {'name': 'left_elbow',
  'id': 7,
  'color': [0, 255, 0],
  'type': 'upper',
  'swap': 'right_elbow'},
 8: {'name': 'right_elbow',
  'id': 8,
  'color': [255, 128, 0],
  'type': 'upper',
  'swap': 'left_elbow'},
 9: {'name': 'left_wrist',
  'id': 9,
  'color': [0, 255, 0],
  'type': 'upper',
  'swap': 'right_wrist'},
 10: {'name': 'right_wrist',
  'id': 10,
  'color': [255, 128, 0],
  'type': 'upper',
  'swap': 'left_wrist'},
 11: {'name': 'left_hip',
  'id': 11,
  'color': [0, 255, 0],
  'type': 'lower',
  'swap': 'right_hip'},
 12: {'name': 'right_hip',
  'id': 12,
  'color': [255, 128, 0],
  'type': 'lower',
  'swap': 'left_hip'},
 13: {'name': 'left_knee',
  'id': 13,
  'color': [0, 255, 0],
  'type': 'lower',
  'swap': 'right_knee'},
 14: {'name': 'right_knee',
  'id': 14,
  'color': [255, 128, 0],
  'type': 'lower',
  'swap': 'left_knee'},
 15: {'name': 'left_ankle',
  'id': 15,
  'color': [0, 255, 0],
  'type': 'lower',
  'swap': 'right_ankle'},
 16: {'name': 'right_ankle',
  'id': 16,
  'color': [255, 128, 0],
  'type': 'lower',
  'swap': 'left_ankle'}}}


global BODYPARTS
BODYPARTS = { 'coco':{
    #"torso": [[0, 6], [6, 5], [5, 11], [11, 12], [12, 6]],
    "torso": [[11, 12]],
    "armr": [[6, 8], [8, 10]],
    "arml": [[5, 7], [7, 9]],
    "legr": [[11, 13], [13, 15]],
    "legl": [[12, 14], [14, 16]]}}



def generate_connectivity_names(connectivity_list, point_names):
    connectivity_names = {}
    for idx, (start, end) in enumerate(connectivity_list):
        start_name = point_names[start]['name']
        end_name = point_names[end]['name']
        connectivity_names[idx] = f"{start_name}_{end_name}"
    return connectivity_names



def get_body_part_vects(pose, connectivity_type='coco'):
    point_info = POINT_INFO[connectivity_type]
    connections = CONNECTIVITY_DICT[connectivity_type]
    connection_names = generate_connectivity_names(connections, point_info)
    
    vects = dict()
    for idx, connection in enumerate(connections):
        vects[connection_names[idx]] = pose[:, connection[1], :] - pose[:, connection[0], :]
    
    return vects


def get_body_part_lengths(pose, connectivity_type='coco'):
    body_part_vects = get_body_part_vects(pose, connectivity_type)
    lengths = dict()
    
    for part in body_part_vects:
        # Check if the vectors are in PyTorch tensors or NumPy arrays
        if isinstance(body_part_vects[part], torch.Tensor):
            lengths[part] = torch.norm(body_part_vects[part], dim=1)
        else:  # Assume it's a NumPy array
            lengths[part] = np.linalg.norm(body_part_vects[part], axis=1)
    
    return lengths










def rotation_conversion(rotation_rep, to_vector = True):
    '''convert between axis-angle vector and matrix representations of roatation'''
    np_type=False
    if type(rotation_rep)==np.ndarray:
        rotation_rep = torch.tensor(rotation_rep)
        np_type=True
    if rotation_rep.shape == (3, 3) and to_vector:  # Convert rotation matrix to axis-angle vector
        R = rotation_rep
        # Calculate the rotation angle
        theta = torch.acos((torch.trace(R) - 1) / 2)
        
        # If theta is close to 0, return a zero vector
        if torch.abs(theta - torch.tensor(0.0))<10**(-6):
            return torch.zeros(3)
        
        # Compute the rotation axis
        ux = (R[2, 1] - R[1, 2]) / (2 * torch.sin(theta))
        uy = (R[0, 2] - R[2, 0]) / (2 * torch.sin(theta))
        uz = (R[1, 0] - R[0, 1]) / (2 * torch.sin(theta))
        
        # Form the axis-angle vector
        R = theta * torch.tensor([ux, uy, uz])

    elif not(rotation_rep.shape == (3, 3) or to_vector):  # Convert axis-angle vector to rotation matrix
        axis_angle = rotation_rep
        theta = torch.norm(axis_angle)
        
        # If theta is close to 0, return the identity matrix
        if torch.abs(theta - torch.tensor(0.0))<10**(-6):
            return torch.eye(3)
        
        # Normalize the axis vector
        u = axis_angle / theta
        ux, uy, uz = u
        
        # Create the skew-symmetric matrix K
        K = torch.tensor([
            [0, -uz, uy],
            [uz, 0, -ux],
            [-uy, ux, 0]
        ])
        
        # Compute the rotation matrix using Rodrigues' formula
        R = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)
    
    else:
        R = rotation_rep
    if np_type:
        R = np.array(R)
    return R


# Function to convert tensor to numpy array if not already a numpy array
def to_numpy(arr):
    return arr if isinstance(arr, np.ndarray) else arr.numpy()



def triangulate_points(kpts_2d, cmtx1, dist1, R1, T1, cmtx2, dist2, R2, T2):
    """
    Triangulate 3D points from 2D keypoints in two camera views.

    Parameters:
    - cmtx1, cmtx2: Camera intrinsic matrices for camera 1 and camera 2
    - dist1, dist2: Distortion coefficients for camera 1 and camera 2
    - R1, R2: Rotation matrices for camera 1 and camera 2
    - T1, T2: Translation vectors for camera 1 and camera 2
    - kpts_2d: 2D keypoints with shape (n_pts, 2, 2), where the last two dimensions are for the cameras and spatial positions respectively

    Returns:
    - points_3d: Triangulated 3D points with shape (n_pts, 3)
    """


    # Convert if not already numpy arrays
    kpts_2d = to_numpy(kpts_2d)
    cmtx1 = to_numpy(cmtx1)
    dist1 = to_numpy(dist1)
    R1 = to_numpy(R1)
    T1 = to_numpy(T1)
    cmtx2 = to_numpy(cmtx2)
    dist2 = to_numpy(dist2)
    R2 = to_numpy(R2)
    T2 = to_numpy(T2)
    



    #make sure input data is stacked
    shape = list(kpts_2d.shape[:-2])
    kpts_2d = kpts_2d.reshape([-1,2,2])
    
    n_pts = kpts_2d.shape[0]

    # Undistort points (points need to be of shape (:,1,2) or (1,:,2) or (1,2))
    kpts_2d_cam1_undistorted = cv.undistortPoints(kpts_2d[:, 0, :][:,None,:], cmtx1, dist1, None, cmtx1)[:,0,:]
    kpts_2d_cam2_undistorted = cv.undistortPoints(kpts_2d[:, 1, :][:,None,:], cmtx2, dist2, None, cmtx2)[:,0,:]

    # Compute projection matrices
    P1 = np.dot(cmtx1, np.hstack((R1, T1.reshape(-1, 1))))
    P2 = np.dot(cmtx2, np.hstack((R2, T2.reshape(-1, 1))))

    # Triangulate points (this can only process 512 points at a time)
    points_4d_homogeneous = []
    start = 0
    for stop in range(512,n_pts+512,512):
        stop = min([stop,n_pts])
        points_4d_homogeneous.append(cv.triangulatePoints(P1, P2, kpts_2d_cam1_undistorted[start:stop].T, kpts_2d_cam2_undistorted[start:stop].T))
        start = stop
    points_4d_homogeneous = np.concatenate(points_4d_homogeneous,-1)
    
    # Convert from homogeneous coordinates to 3D
    points_3d = cv.convertPointsFromHomogeneous(points_4d_homogeneous.T)
#    points_3d.squeeze()
    
    #return to unstacked shape
    points_3d = points_3d.reshape(shape+[3])
    return points_3d





def create_new_numbered_folder(base_dir):
    # Create base dir path if it doesnt already exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Get all integer-named subfolders
    subdirs = [int(d) for d in os.listdir(base_dir) if d.isdigit()]
    
    # Determine the next folder number
    new_folder_num = max(subdirs, default=-1) + 1
    
    # Create the new folder
    new_folder_path = os.path.join(base_dir, str(new_folder_num))
    os.makedirs(new_folder_path)
    
    return new_folder_num







def load_if_exists(path):
    if os.path.exists(path):
        return np.load(path)
    else:
        print(f'file does not exist at path {path}')
        None




#some useful functions for using yaml files to store function parameters
def load_config(config_path=None):
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return config

def get_function_defaults(func):
    sig = inspect.signature(func)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}

def prepare_kwargs(func, user_kwargs):
    default_kwargs = get_function_defaults(func)
    kwargs = default_kwargs.copy()
    kwargs.update(user_kwargs or {})

    # Convert .inf strings back to np.inf
    for k, v in kwargs.items():
        if v == ".inf":
            kwargs[k] = np.inf
        if k == "betas" and isinstance(v, list):
            kwargs[k] = tuple(v)
    return kwargs

