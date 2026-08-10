import numpy as np
import cv2

def load_depth(depth_path, depth_shape=(192, 256), dtype=np.float16) -> np.ndarray:
    """Returns np array of depth data of dtype `dtype` and shape (T, depth_shape[0], depth_shape[1])"""
    # Load the raw file as a numpy array
    with open(depth_path, 'rb') as f:
        depth_data = np.frombuffer(f.read(), dtype=dtype)

    # Reshape the data to the specified shape
    depth_array = depth_data.reshape((-1, *depth_shape))
    return depth_array

def depth_array_to_greyscale_video(depth_array, output_video_path, depth_shape=(192, 256), max_distance=1):
    """Clips the depth array to max_distance and saves it as a greyscale video"""
    depth_video = (depth_array.clip(0,max_distance) / max_distance * 255).astype(np.uint8)

    # Define the output video path and frame dimensions
    frame_width, frame_height = depth_shape
    fps = 60  # Frames per second

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_height, frame_width), isColor=False)

    # Write each frame to the video
    for frame in depth_video:
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

def depth_array_to_color_video(depth_array, output_video_path, depth_shape=(192, 256), max_distance=1):
    """Saves an mp4 video with color map with values up to max_distance. If max_distance is -1, the video will be normalized to the largest value in the array"""
    # Normalize the depth array to the range [0, 255]
    if max_distance == -1:
        max_distance = depth_array.max()
    depth_video = (depth_array.clip(0,max_distance) / max_distance * 255).astype(np.uint8)

    # Define the output video path and frame dimensions
    frame_width, frame_height = depth_shape
    fps = 60  # Frames per second

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_height, frame_width), isColor=True)

    # Apply a color map and write each frame to the video
    for frame in depth_video:
        color_frame = cv2.applyColorMap(frame, cv2.COLORMAP_RAINBOW)
        video_writer.write(color_frame)

    # Release the video writer
    video_writer.release()
