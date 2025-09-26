import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib import gridspec
from matplotlib.patches import Ellipse

import utils
import torch

import argparse
import yaml

from utils import load_frames
def read_frames(recording_path, frame, starting_frame, camera_indices):
    frames = []
    for cam_idx in camera_indices:
        filename = f'frame{frame+starting_frame}.jpg'
        filepath = os.path.join(recording_path, f'camera{cam_idx}', filename)
        if os.path.exists(filepath):
            frame_data = cv.imread(filepath)[:, :, ::-1]
        else:
            # Create a black frame as a placeholder
            height, width = 1080, 1920  # Assuming HD frame dimensions
            frame_data = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(frame_data)
    return frames




def calculate_plot_lims(dat, homogeneuous_lims = True, axis = (0), iqr_margin = 0.5):

        percentile_upper = np.nanpercentile(dat, 95, axis=axis)  # Calculate 95th percentile
        percentile_lower = np.nanpercentile(dat, 5, axis=axis)  # Calculate 5th percentile
        q3 = np.nanpercentile(dat, 75, axis=axis)  # Calculate 75th percentile (Q3)
        q1 = np.nanpercentile(dat, 25, axis=axis)  # Calculate 25th percentile (Q1)
        iqr = q3 - q1  # Calculate IQR

        #ensure that the perccentiles/iqr is are iteratable so we can zip
        if type(percentile_lower) != np.ndarray:
            percentile_lower, percentile_upper, iqr = [[v] for v in [percentile_lower, percentile_upper, iqr]]
        
        # Calculate the lower and upper limits using the 95th percentile + IQR
        lims = [(p5 - iqr_margin * iqr, p95 + iqr_margin * iqr) for p5, p95, iqr in zip(percentile_lower, percentile_upper, iqr)]

        # Recalculate the lims so that they have the same deviation (so the plotted graphs aren't stretched/squished)
        if homogeneuous_lims:
            lim_devs = [lim[1] - lim[0] for lim in lims]
            lim_devs = [max(lim_devs) - ld for ld in lim_devs]
            lims = [(lim[0] - ld / 2, lim[1] + ld / 2) for lim, ld in zip(lims, lim_devs)]
        return lims


def visualize_3d(p3ds, body_parts, additional_metrics=[], additional_metric_names=[], point_labels=[], recording_paths=None, n_frames=None, camera_indices=None, starting_point=0, starting_frame=None, plane_views=['xy', 'zy', 'zx']):
    """
    Visualizes 3D body part movements and optional additional metrics over time using Matplotlib.

    Parameters:
    -----------
    p3ds : ndarray
        Array of 3D coordinates of shape (n_frames, n_points, 3), representing the 3D positions of body points over time.
    body_parts : dict
        A dictionary where each key corresponds to a body part, and each value is a list of tuples defining point indices that should be connected for that body part.
    additional_metrics : list of ndarray, optional
        List of additional metrics (e.g., 2D points or time-series data) to visualize, each with shape (n_frames, n_points, ...) depending on the type of data.
    additional_metric_names : list of str, optional
        List of names corresponding to each additional metric. If fewer names than metrics are provided, the function auto-generates names.
    point_labels : list of str, optional
        Labels for each point. Used when visualizing additional metrics. If fewer labels than points are provided, the function pads with empty strings.
    recording_paths : dict, optional
        Dict of files of recorded video frames for the cameras, if visualizing frames from specific cameras.
    n_frames : int, optional
        Number of frames to visualize. If None, it defaults to the total number of frames in p3ds minus the starting frame.
    camera_indices : list of int, optional
        Indices of the cameras to visualize. If None, defaults to cameras 0 and 1.
    starting_point : int, optional
        Point from which to start the visualization. Defaults to 0.
    starting_frame : int, optional
        Frame from which to start the video visualization (often larger than starting_point since p3ds often isnt of entire video). Defaults to 0.
    plane_views : list of str, optional
        List of plane views to visualize (e.g., 'xy', 'zy', 'zx'). Defaults to ['xy', 'zy', 'zx'].

    Returns:
    --------
    ani : matplotlib.animation.FuncAnimation
        Animation object displaying the 3D body parts and additional metrics over time.

    Notes:
    ------
    - If `additional_metrics` is empty, the function will visualize only the 3D body parts.
    - Handles auto-generation of additional metric names and point labels when their lists are shorter than expected.
    """

    p3ds[:,:,1] *=-1


    if starting_frame is None:
        starting_frame = starting_point

    if n_frames is None:
        n_frames = len(p3ds) - starting_frame

    
    if camera_indices is None:
        camera_indices = [0, 1]

    # Ensure additional metric names are sufficient
    if len(additional_metric_names) < len(additional_metrics):
        additional_metric_names += [f'additional_metric{i}' for i in range(len(additional_metrics) - len(additional_metric_names))]

    # Ensure point labels are sufficient if additional metrics are provided
    if additional_metrics and len(point_labels) < additional_metrics[0].shape[1]:
        point_labels += [f'point{i}' for i in range(additional_metrics[0].shape[1] - len(point_labels))]

    body = body_parts.values()
    colors = ['red', 'blue', 'green', 'black', 'orange']
    
        # Count how many subplots we need
    n_plane_views = len(plane_views)
    n_cams = len(camera_indices) if recording_paths else 0
    n_metrics = len(additional_metrics)
    total_plots = n_plane_views + n_cams + n_metrics
    
    # Calculate grid dimensions
    n_cols = max(n_plane_views, n_cams, n_metrics, 1)
    n_rows = sum([n_plane_views > 0, n_cams > 0, n_metrics > 0])
    
    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    axes = {}
    all_lims = {}
    
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    row = 0
    # Plane view subplots
    for i, plane_view in enumerate(plane_views):
        axes[plane_view] = fig.add_subplot(gs[row, i], projection='3d')
        
        #calculate plot limits only using the points being displayed
        displayed_indecies = list(set(sum([sum(v,[]) for v in utils.BODYPARTS['coco'].values()], [])))
        all_lims[plane_view] = calculate_plot_lims(p3ds[:,displayed_indecies,:], axis=(0, 1))
    row += 1 if n_plane_views > 0 else 0
    
    # Camera view subplots
    for i, cam_idx in enumerate(camera_indices if recording_paths else []):
        axes[f'cam{i}'] = fig.add_subplot(gs[row, i])
    row += 1 if n_cams > 0 else 0
    
    # Additional metric subplots
    for i, additional_metric in enumerate(additional_metrics):
        axes[additional_metric_names[i]] = fig.add_subplot(gs[row, i])
        all_lims[additional_metric_names[i]] = calculate_plot_lims(additional_metric[starting_point:n_frames+starting_point], axis=(1))
    all_frames = load_frames(recording_paths, start_end_frames=[starting_frame+1, n_frames+starting_frame])

    def update(frame):

        frame = frame + starting_point
        if all_frames is None or frame==starting_point:

            height, width = 1080, 1920  # Assuming HD frame dimensions
            frame_data = np.zeros((height, width, 3), dtype=np.uint8)
            frames = [frame_data] * len(camera_indices)
        else:
            frames = next(all_frames)
        for plane_name, ax in axes.items():
            ax.cla()
            try:
                lims = all_lims[plane_name]
            except KeyError:
                pass

            labels = ['x', 'y', 'z']
            if 'cam' in plane_name:
                ax.imshow(frames[int(plane_name[3:])])
                ax.axis('off')
            elif plane_name in additional_metric_names:
                idx = additional_metric_names.index(plane_name)
                additional_metric = additional_metrics[idx]

                if len(additional_metric.shape) == 2:
                    #ax.set_aspect('equal')

                    interval_size = 30
                    for i in range(additional_metric.shape[1]):
                        ax.plot(range(starting_point, len(additional_metric[:, i])), additional_metric[starting_point:, i], label=point_labels[i])
                    ax.set_xlabel('Time step')
                    ax.set_ylabel(plane_name)
                    ax.set_title(f'{plane_name} over time')
                    ax.set_ylim(lims[0])
                    ax.set_xlim([frame - interval_size, frame])
                else:
                    for i in range(additional_metric.shape[1]):
                        ax.scatter(additional_metric[frame, i, 0], additional_metric[frame, i, 1], label=point_labels[i], marker='o')
                    ax.set_title(f'{plane_name}')
                    ax.set_xlim(lims[0])
                    ax.set_ylim(lims[1])

                if point_labels and plane_name == additional_metric_names[0]:
                    ax.legend(fontsize=6, markerscale=0.5, loc='lower left')
            else:
                if plane_name == 'xy':
                    ax.view_init(elev=90, azim=-90)
                    xyz_indices = [0, 1, 2]
                elif plane_name == 'zy':
                    ax.view_init(elev=0, azim=0)
                    xyz_indices = [0, 2, 1]
                elif plane_name == 'zx':
                    ax.view_init(elev=-90, azim=0)
                    xyz_indices = [2, 0, 1]

                xlabel, ylabel, zlabel = [labels[i] for i in xyz_indices]
                xlim, ylim, zlim = [lims[i] for i in xyz_indices]

                for bodypart, part_color in zip(body, colors):
                    for _c in bodypart:
                        xyz = [[p3ds[frame][_c[0], xyz_indices[0]], p3ds[frame][_c[1], xyz_indices[0]]],
                               [p3ds[frame][_c[0], xyz_indices[1]], p3ds[frame][_c[1], xyz_indices[1]]],
                               [p3ds[frame][_c[0], xyz_indices[2]], p3ds[frame][_c[1], xyz_indices[2]]]]
                        if not (max([xyz[i][0] is np.nan for i in range(3)])):
                            ax.plot(*xyz, linewidth=4, c=part_color)
                ax.set_xlim3d(*xlim)
                ax.set_ylim3d(*ylim)
                ax.set_zlim3d(*zlim)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_zlabel(zlabel)
                ax.set_title(f'{plane_name.upper()} Plane')

    ani = FuncAnimation(fig, update, frames=range(n_frames), interval=100)
    return ani

        

def create_heatmap_animation(heatmaps, recording_paths, starting_frame = 0, n_frames = None, output_path='animation.gif'):
    """
    Creates an animated GIF overlaying Gaussian heatmaps on video frames using matplotlib.animation.FuncAnimation.

    Parameters:
    - heatmaps (np.ndarray): Tensor of shape (time, n_cams, n_points, 6), containing Gaussian parameters.
                             The last dimension includes the mean (x, y) and flattened covariance matrix.
    - all_frames (generator): Yields lists of length n_cams, each containing video frames (np.ndarray of shape (width, height, 3)).
    - output_path (str): Path to save the resulting animated GIF.
    """
    time, n_cams, n_points, _ = heatmaps.shape
    fig, axes = plt.subplots(1, n_cams, figsize=(5 * n_cams, 5))
    if n_frames is None:
        n_frames = time
    all_frames = load_frames(recording_paths, start_end_frames=[starting_frame+1, n_frames+starting_frame])

    
    if n_cams == 1:
        axes = [axes]  # Ensure axes is iterable for single-camera case

    image_artists = []
    ellipse_artists = [[] for _ in range(n_cams)]

    # Initialize the plots for each camera
    for cam_idx in range(n_cams):
        axes[cam_idx].axis('off')
        image_artist = axes[cam_idx].imshow(np.zeros((1, 1, 3), dtype=np.uint8))  # Placeholder image
        image_artists.append(image_artist)

        for _ in range(n_points):
            ellipse = Ellipse((0, 0), 0, 0, edgecolor='red', fill=False)
            axes[cam_idx].add_patch(ellipse)
            ellipse_artists[cam_idx].append(ellipse)

    # Helper function to update ellipses for the covariance matrix
    def update_ellipses(mean, cov_matrix, ellipse):
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(eigvals)
        ellipse.set_center(mean)
        ellipse.width = width
        ellipse.height = height
        ellipse.angle = angle

    def update(frame_data):
        frames, heatmaps_t = frame_data

        for cam_idx in range(n_cams):
            # Update video frame
            image_artists[cam_idx].set_data(frames[cam_idx])

            # Update heatmap overlays
            cam_heatmaps = heatmaps_t[cam_idx]
            for point_idx, (mean, ellipse) in enumerate(zip(cam_heatmaps[:, :2], ellipse_artists[cam_idx])):
                cov = cam_heatmaps[point_idx, 2:].reshape(2, 2)
                update_ellipses(mean, cov, ellipse)

    # Generator to supply the animation frames
    def data_generator():
        for t, frames in enumerate(all_frames):
            yield frames, heatmaps[t]

    anim = FuncAnimation(
        fig, update, frames=data_generator, interval=100, blit=False
    )

    return anim



def overlay_heatmap(ax, frame, heatmaps, n_points):
    """
    Overlays heatmap points and ellipses onto an image.

    Parameters:
    - ax (matplotlib.axes.Axes): The axis to draw on.
    - frame (np.ndarray): Video frame to display as the background (shape: (height, width, 3)).
    - heatmaps (np.ndarray): Heatmap for this frame (shape: (n_points, 6)).
    - n_points (int): Number of points in the heatmap.

    Returns:
    - list: List of ellipse artists added to the axis.
    """
    # Display the video frame
    ax.imshow(frame)
    ax.axis('off')

    ellipse_artists = []
    for i in range(n_points):
        mean = heatmaps[i, :2]  # Mean (x, y)
        cov_matrix = heatmaps[i, 2:].reshape(2, 2)  # Covariance matrix [[a, b], [b, c]]

        # Check if covariance matrix is valid
        if np.linalg.det(cov_matrix) <= 0:
            print(f"Invalid covariance matrix for point {i}: {cov_matrix}")
            continue

        # Compute ellipse properties
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(eigvals)

        # Ensure eigenvalues are non-negative
        if width <= 0 or height <= 0:
            print(f"Skipping point {i} due to non-positive dimensions: width={width}, height={height}")
            continue

        # Create ellipse
        ellipse = Ellipse(mean, width, height, angle = angle, edgecolor='red', fill=False, lw=0.5)
        ax.add_patch(ellipse)
        ellipse_artists.append(ellipse)

        # Plot the center point
        ax.plot(mean[0], mean[1], 'ro', markersize=2)

    return ellipse_artists



def heatmap_animation(
    heatmaps, recording_paths, starting_frame=0, n_frames=None
):
    """
    Creates an animated GIF overlaying Gaussian heatmaps on video frames using matplotlib.animation.FuncAnimation.

    Parameters:
    - heatmaps (np.ndarray): Tensor of shape (time, n_cams, n_points, 6), containing Gaussian parameters.
    - recording_paths (list): List of paths to video recordings for each camera.
    - starting_frame (int): Frame index to start processing.
    - n_frames (int): Number of frames to process. If None, processes all frames.
    - output_path (str): Path to save the resulting animated GIF.
    """
    time, n_cams, n_points, _ = heatmaps.shape
    if n_frames is None:
        n_frames = time

    all_frames = load_frames(
        recording_paths, start_end_frames=[starting_frame, n_frames + starting_frame-1]
    )

    fig, axes = plt.subplots(1, n_cams, figsize=(5 * n_cams, 5))
    if n_cams == 1:
        axes = [axes]  # Ensure axes is iterable for single-camera case

    # Initialize the plots for each camera
    image_artists = []
    ellipse_artists = [[] for _ in range(n_cams)]
    for cam_idx in range(n_cams):
        axes[cam_idx].axis('off')
        image_artist = axes[cam_idx].imshow(np.zeros((1, 1, 3), dtype=np.uint8))  # Placeholder
        image_artists.append(image_artist)

    def update(frame_data):
        frames, heatmaps_t = frame_data

        for cam_idx in range(n_cams):
            # Update the frame for the camera
            frame = frames[cam_idx]
            heatmap = heatmaps_t[cam_idx]
            ax = axes[cam_idx]
            ax.clear()
            ellipse_artists[cam_idx] = overlay_heatmap(ax, frame, heatmap, n_points)

    # Generator to supply frames
    def data_generator():
        for t, frames in enumerate(all_frames):
            yield frames, heatmaps[t]

    anim = FuncAnimation(
        fig, update, frames=data_generator, interval=100, blit=False
    )


    return anim

def interactive_3d_pose_animation(p3ds, body_parts):
    """
    Creates an interactive 3D plot to visualize body movements over time,
    allowing for control via sliders for azimuth, elevation, and roll.
    
    Parameters:
    -----------
    p3ds : ndarray
        Array of 3D coordinates of shape (n_frames, n_points, 3), representing the 3D positions of body points over time.
    body_parts : dict
        A dictionary where each key corresponds to a body part, and each value is a list of tuples defining point indices 
        that should be connected for that body part.

    Returns:
    --------
    None
        Displays an interactive animation of the 3D poses in a new window.
    """
    # Setup figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits based on p3ds data
    x_range = np.max(p3ds[:, :, 0]) - np.min(p3ds[:, :, 0])
    y_range = np.max(p3ds[:, :, 1]) - np.min(p3ds[:, :, 1])
    z_range = np.max(p3ds[:, :, 2]) - np.min(p3ds[:, :, 2])
    
    max_range = max(x_range, y_range, z_range)

    ax.set_xlim(np.min(p3ds[:, :, 0]), np.min(p3ds[:, :, 0]) + max_range)
    ax.set_ylim(np.min(p3ds[:, :, 1]), np.min(p3ds[:, :, 1]) + max_range)
    ax.set_zlim(np.min(p3ds[:, :, 2]), np.min(p3ds[:, :, 2]) + max_range)

    # Initialize azimuth, elevation, and roll
    azm = 270
    elev = 90
    roll = 0

    # Line drawing colors
    colors = ['red', 'blue', 'green', 'black', 'orange']
    body = list(body_parts.values())

    # Initial empty lines for each point pair in body_parts
    lines = []
    for bodypart, part_color in zip(body, colors):
        for point_pair in bodypart:
            line, = ax.plot([], [], [], c=part_color, linewidth=2)
            lines.append(line)

    # Update function for animation
    def update(frame):
        line_idx = 0
        for bodypart in body:
            for point_pair in bodypart:
                p1, p2 = point_pair
                x_vals = [p3ds[frame, p1, 0], p3ds[frame, p2, 0]]
                y_vals = [p3ds[frame, p1, 1], p3ds[frame, p2, 1]]
                z_vals = [p3ds[frame, p1, 2], p3ds[frame, p2, 2]]
                lines[line_idx].set_data(x_vals, y_vals)
                lines[line_idx].set_3d_properties(z_vals)
                line_idx += 1

        # Update the view
        ax.view_init(elev, azm, roll)

    # Slider setup
    axcolor = 'lightgoldenrodyellow'
    ax_azm = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor=axcolor)
    ax_elev = plt.axes([0.1, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_roll = plt.axes([0.1, 0.09, 0.65, 0.03], facecolor=axcolor)

    s_azm = Slider(ax_azm, 'Azimuth', 0, 360, valinit=azm)
    s_elev = Slider(ax_elev, 'Elevation', 0, 180, valinit=elev)
    s_roll = Slider(ax_roll, 'Roll', 0, 360, valinit=roll)

    # Update the azimuth, elevation, and roll when sliders are changed
    def update_view(val):
        nonlocal azm, elev, roll
        azm = s_azm.val
        elev = s_elev.val
        roll = s_roll.val

    s_azm.on_changed(update_view)
    s_elev.on_changed(update_view)
    s_roll.on_changed(update_view)

    # Animation using FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(p3ds), interval=100, blit=False)

    # Open in a new window and keep the animation running
    plt.show()

    
    
    
    
    
    
    
    
    
def overlay_trackpoints(keypoints, img, keypoint_info, print_info = False, print_uncertain = False, confidence_threshold = 0.9):
    
    # Get keypoints for the current image
    num_keypoints = len(keypoints['points'])

    # Loop over each keypoint
    for i in range(num_keypoints):
        confidence = keypoints['confidence'][i]
        if print_uncertain or confidence >= confidence_threshold:
            if print_info:
                print(f"Point {i} {keypoint_info[i]['name']} confidence: {keypoints['confidence'][i]}")
    
            if confidence < confidence_threshold:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            name = keypoint_info[i].get('name', '')
    
            cv.circle(img, (int(keypoints['points'][i][0]), int(keypoints['points'][i][1])), 5, color, -1)
            cv.putText(img, name, (int(keypoints['points'][i][0]) + 10, int(keypoints['points'][i][1]) + 10),
                        cv.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3, cv.LINE_AA)






def animate_trackpoints(frames, kpts_2d, keypoint_info, print_info=False, print_uncertain=False, confidence_threshold=0.9):
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(cv.cvtColor(frames[0], cv.COLOR_BGR2RGB))

    def update(frame_idx):
        img = frames[frame_idx].copy()
        keypoints = {'points': kpts_2d[frame_idx], 'confidence': [1.0] * len(kpts_2d[frame_idx])}
        overlay_trackpoints(keypoints, img, keypoint_info, print_info, print_uncertain, confidence_threshold)
        im.set_array(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        return im,

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    plt.show()
    return ani




def visualize_2d(kpts, body_parts, draw_lines=True):
    n_frames, n_cameras, n_points, _ = kpts.shape

    # Create a figure and axis for the plot
    fig, axs = plt.subplots(1, n_cameras, figsize=(n_cameras * 8, 8))

    # Create a scatter plot for each camera
    scatters = []
    colors = ['red', 'blue', 'green', 'black', 'orange']
    for i, ax in enumerate(axs):
        # Get the x and y coordinates for all frames and points for the current camera
        x = kpts[:, i, :, 0].reshape(-1).numpy()
        y = kpts[:, i, :, 1].reshape(-1).numpy()

        # Compute the x and y limits based on the min/max values of x and y
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_margin = 0.1 * x_range
        y_margin = 0.1 * y_range

        # Set the x and y limits with a small margin
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_title(f'Camera {i}')
        ax.grid(True)
        scatters.append(ax.scatter([], [], color='blue'))

    # Function to update the scatter plots for each frame
    def update(frame):
        for i, scatter in enumerate(scatters):
            ax = scatter.axes
            x = kpts[frame, i, :, 0].numpy()  # Get x coordinates
            y = kpts[frame, i, :, 1].numpy()  # Get y coordinates
            offsets = np.vstack((x, y)).T  # Convert to numpy array and transpose
            scatter.set_offsets(offsets)  # Set new offsets for scatter plot
    
            if draw_lines:
                for line in ax.lines:
                    line.remove()  # Clear previous lines
                for bodypart, part_color in zip(body_parts.values(), colors):
                    for _c in bodypart:
                        x = kpts[frame, i, _c, 0].numpy()
                        y = kpts[frame, i, _c, 1].numpy()
                        if not np.isnan(x).any() and not np.isnan(y).any():
                            ax.plot(x, y, linewidth=4, c=part_color)
    
            else:
                for idx, (_x, _y) in enumerate(zip(x, y)):
                    ax.annotate(str(idx), (_x, _y), textcoords="offset points", xytext=(0,10), ha='center')
    
        return scatters

    # Create the animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)

    return anim

        







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recording_log', type=str, help='Path to recording log')
    parser.add_argument('--heatmaps_2d', type=str, help='Path to 2D heatmaps .npy file')
    parser.add_argument('--kpts_2d', type=str, help='Path to 2D keypoints .npy file')
    parser.add_argument('--kpts_3d', type=str, help='Path to 3D keypoints .npy file')
    parser.add_argument('--estimator_model', type=str, help='Name of the estimator model used (e.g. coco_base)')
    parser.add_argument('--recording_paths', nargs='+', help='Paths to synced recording video files')
    parser.add_argument('--plot_types', nargs='+', help='Type(s) of plot to create ("heatmap" or "3D_pose")')
    parser.add_argument('--save_plots', action='store_true', default=True)
    parser.add_argument('--save_path', type=str, help='Path in which to save plots. Will default to recording_log path if available, and otherwise the current directory.')
    
    args = parser.parse_args()

    if args.plot_types is None:
        args.plot_types = ['heatmap']


    #get save directory
    if args.save_path is None:
        if args.recording_log is not None:
            args.save_path = os.path.dirname(args.recording_log)
        else:
            args.save_path = os.getcwd()
    
    
    if args.recording_log is not None:
        with open(args.recording_log) as f:
            log = yaml.safe_load(f)
    
    
    for arg_name, arg_value in vars(args).items():
        if arg_value is None:
            setattr(args, arg_name, log[arg_name])
    
    
    #load data
    kpts_3d = utils.load_if_exists(args.kpts_3d) 
    kpts_2d = utils.load_if_exists(args.kpts_2d)
    heatmaps = utils.load_if_exists(args.heatmaps_2d)
    heatmaps = torch.tensor(heatmaps)
    recording_paths = args.recording_paths
    estimator_model = args.estimator_model
    save_path = args.save_path    
    
    #construct plots
    anis = dict()
    for plot_type in args.plot_types:
        if plot_type == 'heatmap':
            ani = heatmap_animation(heatmaps,recording_paths)
            
        elif plot_type == '3D_pose':
            #load body parts for plotting 3d pose
            key_name = ''
            if 'coco' in estimator_model:
                key_name = 'coco'
            body_parts = utils.BODYPARTS[key_name]

            ani = visualize_3d(kpts_3d, body_parts, recording_paths=recording_paths)
        else:
            raise ValueError(f'plot_type "{plot_type}" is invalid! Must be one of the following: "heatmap", "3D_pose"')
        
        anis[plot_type] = ani
    if args.save_plots:
        for plot_type, ani in anis.items():
            #if the path exists, assume the save_path is for the folder and otherwise assume its for the path+filename
            if os.path.exists(save_path):
                ani_save_path = os.path.join(save_path, f'{plot_type}.gif')
            else:
                ani_save_path = save_path+f'_{plot_type}.gif'
            print(f'saving animation {plot_type} at path {ani_save_path}')
            ani.save(ani_save_path, fps=10)
    
    
    
