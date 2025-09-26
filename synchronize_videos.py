import os
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import tempfile
import cv2
import utils
import math
import platform
import subprocess

def get_loudest_point(audio_path):
    # Load audio file with librosa
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract the first 10 seconds
    y_10s = y[:sr * 30]
    
    # Find the loudest point in the first 10 seconds
    loudest_index = np.argmax(np.abs(y_10s))
    return loudest_index / sr  # Convert sample index to time in seconds






def display_frame_grid(reference_frame, video_frames, frame_ranges):
    grid_size = (len(video_frames) + 1, max(len(range_) for range_ in frame_ranges))
    frame_height, frame_width, _ = reference_frame.shape
    grid_image = np.zeros((frame_height * grid_size[0], frame_width * grid_size[1], 3), dtype=np.uint8)

    # Display the reference frame
    grid_image[:frame_height, :frame_width] = reference_frame
    cv2.putText(grid_image, "Reference", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Arrange frames from other videos in rows
    for video_idx, (frames, frame_range) in enumerate(zip(video_frames, frame_ranges)):
        for idx, (frame_idx, frame) in enumerate(zip(frame_range, frames)):
            row = video_idx + 1
            col = idx
            grid_image[row*frame_height:(row+1)*frame_height, col*frame_width:(col+1)*frame_width] = frame
            label_position = (col * frame_width + 10, row * frame_height + 50)
            cv2.putText(grid_image, str(frame_idx), label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Select Matching Frames", grid_image)
    cv2.waitKey(1)

    selected_frame_indices = []
    for i, frame_range in enumerate(frame_ranges):
        selected_frame_idx = int(input(f"Enter the integer position {frame_range[0]}-{frame_range[-1]} for video {i+1}: "))
        selected_frame_indices.append(selected_frame_idx)

    utils.destroy_windows_mac()
    return selected_frame_indices

def create_split_screen(frames_list, output_path, fps=30):
    if len(set(len(frames) for frames in frames_list)) != 1:
        raise ValueError("All videos must have the same number of frames.")

    height, width, _ = frames_list[0][0].shape
    output_size = (width * len(frames_list), height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

    for frame_set in zip(*frames_list):
        combined_frame = np.hstack(frame_set)
        out.write(combined_frame)

    out.release()
    
    
    
    
    
def create_scrollable_grid(video_frames, frame_ranges, scale_factor=0.5):
    num_videos = len(video_frames)
    max_frames = max(len(range_) for range_ in frame_ranges)
    frame_height, frame_width, _ = video_frames[0][0].shape

    # Calculate scaled dimensions
    scaled_height = int(frame_height * scale_factor)
    scaled_width = int(frame_width * scale_factor)

    # Create a large grid image
    grid_image = np.zeros((num_videos * scaled_height, max_frames * scaled_width, 3), dtype=np.uint8)

    for video_idx, (frames, frame_range) in enumerate(zip(video_frames, frame_ranges)):
        for idx, (frame_idx, frame) in enumerate(zip(frame_range, frames)):
            y_start = video_idx * scaled_height
            x_start = idx * scaled_width
            
            # Resize the frame
            resized_frame = cv2.resize(frame, (scaled_width, scaled_height))
            
            grid_image[y_start:y_start+scaled_height, x_start:x_start+scaled_width] = resized_frame
            
            # Add text label
            cv2.putText(grid_image, f"V{video_idx+1}: {frame_idx}", 
                        (x_start + 10, y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, (255, 0, 0), 1)

    return grid_image


def create_single_camera_grid(frames, frame_range, rows, cols):
    frame_height, frame_width, _ = frames[0].shape
    grid_height = rows * frame_height
    grid_width = cols * frame_width
    
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for idx, (frame_idx, frame) in enumerate(zip(frame_range, frames)):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        y_start = row * frame_height
        x_start = col * frame_width
        
        grid_image[y_start:y_start+frame_height, x_start:x_start+frame_width] = frame
        
        # Add text label
        cv2.putText(grid_image, f"Frame: {frame_idx}", 
                    (x_start + 10, y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return grid_image

def bring_cv2_window_to_front(window_name):
    if platform.system() == "Darwin":  # macOS only
        # Use AppleScript to bring Python (or Terminal) to front
        script = '''
        tell application "System Events"
            set frontmost of the first process whose name is "Python" to true
        end tell
        '''
        subprocess.call(["osascript", "-e", script])

def display_and_select_frame(video_frames, frame_ranges):
    window_name = "Select Synchronization Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1920, 1080)
    cv2.moveWindow(window_name, 50, 50)  # Move near top-left

    selected_frame_indices = []

    for camera_idx, (frames, frame_range) in enumerate(zip(video_frames, frame_ranges)):
        total_frames = len(frames)
        cols = math.ceil(math.sqrt(total_frames))
        rows = math.ceil(total_frames / cols)

        # Display all frames in one grid
        grid_image = create_single_camera_grid(frames, frame_range, rows, cols)

        # Large header and instructions
        cv2.putText(grid_image, f"Camera {camera_idx + 1}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        cv2.putText(grid_image, "Determine the synchronization point and then press Enter", 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(grid_image, "to enter the corresponding frame number in the terminal.",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Overlay frame numbers
        for idx, frame_num in enumerate(frame_range):
            row = idx // cols
            col = idx % cols
            x = col * (grid_image.shape[1] // cols) + 100
            y = row * (grid_image.shape[0] // rows) + 300
            cv2.putText(grid_image, f"{frame_num}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)

        # Show window
        cv2.imshow(window_name, grid_image)
        bring_cv2_window_to_front(window_name)
        cv2.waitKey(1)  # Ensure display

        # Prompt user
        input("Press Enter when ready to enter the frame number in the terminal...")
        while True:
            try:
                selected_idx = int(input(f"Enter the frame number for camera {camera_idx + 1}: "))
                if frame_range[0] <= selected_idx <= frame_range[-1]:
                    selected_frame_indices.append(selected_idx)
                    break
                else:
                    print(f"Frame number must be between {frame_range[0]} and {frame_range[-1]}. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")

    cv2.destroyAllWindows()
    return selected_frame_indices


adjusted_sync_frame_indices = [321, 419, 193]

def synchronize_videos(video_paths, frame_range=list(range(-5, 6)), save_as_files=True, adjusted_sync_frame_indices = None, delete_originals = False):
    audio_paths = [tempfile.mktemp(suffix=".wav") for _ in video_paths]
    output_paths = None
    
    for video_path, audio_path in zip(video_paths, audio_paths):
        VideoFileClip(video_path).audio.write_audiofile(audio_path)

    loudest_times = [get_loudest_point(audio_path) for audio_path in audio_paths]
    videos = [VideoFileClip(path) for path in video_paths]
    fps_list = [video.fps for video in videos]
    sync_frame_indices = [int(time * fps) for time, fps in zip(loudest_times, fps_list)]

    caps = [cv2.VideoCapture(path) for path in video_paths]

    if adjusted_sync_frame_indices is None:   
        all_frames = []
        frame_ranges = []
        for cap, sync_frame_idx in zip(caps, sync_frame_indices):
            frame_indices = [sync_frame_idx + idx for idx in frame_range]
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            all_frames.append(frames)
            frame_ranges.append(frame_indices)
         
        adjusted_sync_frame_indices = display_and_select_frame(all_frames, frame_ranges)
    
    if adjusted_sync_frame_indices is None:
        print("Frame selection cancelled.")
        return None
    
    total_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    overlap_frames = min(total - start for total, start in zip(total_frames, adjusted_sync_frame_indices))

    synchronized_frames = []
    if save_as_files:
        output_paths = [os.path.join(os.path.dirname(path), 
                                     os.path.splitext(os.path.basename(path))[0] + "_synced.mp4") 
                        for path in video_paths]
        writers = [cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                                   (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                   for path, fps, cap in zip(output_paths, fps_list, caps)]

    for cap, start_frame in zip(caps, adjusted_sync_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    #get video FPS in order to maintain synchronization in the case of a range of values
    cap_FPS = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
    adjustment_rates = [max(cap_FPS)/(max(cap_FPS)-FPS) if max(cap_FPS)!=FPS else np.inf for FPS in cap_FPS]
    adjustments_made = [0 for cap in caps]
    previous_frames = []
    for frame_idx in range(overlap_frames):
        frames = []
        for i, cap in enumerate(caps):
            if frame_idx >= (adjustments_made[i]+1)*adjustment_rates[i]:
                frame = previous_frames[i]
                adjustments_made[i]+=1
                ret=True
            else:
                ret, frame = cap.read()
            if ret:
                frames.append(frame)
                if save_as_files:
                    writers[i].write(frame)
            else:
                break
        if len(frames) == len(caps):
            synchronized_frames.append(frames)
        else:
            break
        previous_frames = frames

    for cap in caps:
        cap.release()
    if save_as_files:
        for writer in writers:
            writer.release()


    if delete_originals:
        for video_path in video_paths:
            os.remove(video_path)


    return synchronized_frames, output_paths
    
    


def load_video_frames(video_paths):
    """
    Load frames from multiple MP4 video files.
    
    Parameters:
        video_paths (list): List of paths to video files.
        
    Returns:
        all_frames (list of lists): List containing lists of frames for each video.
    """
    all_frames = []

    for video_path in video_paths:
        # Create a VideoCapture object for each video
        cap = cv2.VideoCapture(video_path)

        # Check if the video file opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            continue

        # List to hold the frames for the current video
        frames = []

        # Read and store the frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Release the VideoCapture object
        cap.release()

        # Add the frames of the current video to all_frames
        all_frames.append(frames)

    return all_frames

