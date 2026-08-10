import os
from umi_day.common.trajectory_util import plot_video_aligned_trajectory
from umi_day.common.transform_util import pos_quat_xyzw_to_4x4
from umi_day.demonstration_processing.utils.generic_util import demonstration_to_display_string, get_demonstration_json_data, get_demonstration_sides_present
from umi_day.demonstration_processing.utils.gripper_util import get_demo_gripper_width, iphone_to_tcp_poses
from umi_day.common.plot_util import plot_gripper_width
from umi_day.demonstration_processing.utils.depth_util import load_depth, depth_array_to_color_video
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def visualize_iphone_data(demonstration_iterator, cfg):
    num_processed = 0
    num_already_processed = 0
    for demonstration_dir in demonstration_iterator('demonstration'):
        def visualize_side(side):
            nonlocal num_processed, num_already_processed

            demonstration_json = get_demonstration_json_data(demonstration_dir, side)
            poses = np.array(demonstration_json['poseTransforms'])
            
            video_out_path = f'{demonstration_dir}/{side}_visualized.mp4'
            if os.path.exists(video_out_path) and not cfg.overwrite:
                num_already_processed += 1
                return
            print(f'{demonstration_to_display_string(demonstration_dir, side)} Beginning to visualize on {side} side')

            # Convert the depth data into color
            depth_path = os.path.join(demonstration_dir, f'{side}_depth.raw')
            depth_array = load_depth(depth_path, depth_shape=(192, 256), dtype=np.float16)
            depth_color_video_path = os.path.join(demonstration_dir, f'{side}_depth_color.mp4')
            depth_array_to_color_video(depth_array, depth_color_video_path, depth_shape=(192, 256), max_distance=-1)
            clipped_depth_color_video_path = os.path.join(demonstration_dir, f'{side}_depth_color_clipped.mp4')
            depth_array_to_color_video(depth_array, clipped_depth_color_video_path, depth_shape=(192, 256), max_distance=cfg.depth_max_distance)
            
            # Visualize the main camera RGB and poses
            tasks_video_path = os.path.join(demonstration_dir, f'subtasks.mp4')
            if os.path.exists(tasks_video_path):
                wide_rgb_video_path = tasks_video_path    
            else:
                wide_rgb_video_path = os.path.join(demonstration_dir, f'{side}_rgb.mp4')
            is_lost = np.zeros(poses.shape[0], dtype=bool)
            intermediate_video_out_path = f'{demonstration_dir}/{side}_visualized_intermediate.mp4'
            tcp_poses = iphone_to_tcp_poses(poses)
            plot_video_aligned_trajectory(wide_rgb_video_path, intermediate_video_out_path, tcp_poses, is_lost, max_frames=cfg.visualize_til_frame)

            # Visualize the ultrawide RGB and depth
            ultrawide_rgb_video_path = os.path.join(demonstration_dir, f'{side}_ultrawidergb.mp4')

            # Load the ultrawide RGB and depth videos
            ultrawide_cap = cv2.VideoCapture(ultrawide_rgb_video_path)
            depth_cap = cv2.VideoCapture(depth_color_video_path)
            clipped_depth_cap = cv2.VideoCapture(clipped_depth_color_video_path)
            intermediate_cap = cv2.VideoCapture(intermediate_video_out_path)

            # Get video properties
            width = int(intermediate_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
            height = int(intermediate_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = intermediate_cap.get(cv2.CAP_PROP_FPS)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
            out = cv2.VideoWriter(video_out_path, fourcc, fps, (width * 3, height * 2))

            # Ultrawide is only at 10Hz, so only read frames when a new frame appears
            ultrawide_timestamps = demonstration_json['ultrawideRGBTimes']
            ultrawide_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Gripper width
            gripper_widths, gripper_detection_types = get_demo_gripper_width(demonstration_dir, side, include_detection_types=True)

            for frame_i in tqdm(range(poses.shape[0]), total=poses.shape[0], leave=False, desc='combining videos'):
                if ultrawide_timestamps[frame_i] != "":
                    ret1, ultrawide_frame = ultrawide_cap.read()
                ret2, depth_frame = depth_cap.read()
                ret3, clipped_depth_frame = clipped_depth_cap.read()
                ret4, intermediate_frame = intermediate_cap.read()

                if not ret2 or not ret3 or not ret4:
                    break

                # Resize depth and ultra wide
                depth_frame = cv2.resize(depth_frame, (width, height))
                clipped_depth_frame = cv2.resize(clipped_depth_frame, (width, height))
                ultrawide_frame = cv2.resize(ultrawide_frame, (width, height))

                # Horizontally stack ultrawide and depth frames
                horizontal_stack = np.hstack((ultrawide_frame, depth_frame))

                # Vertically stack with intermediate frame
                vertical_stack = np.vstack((intermediate_frame, horizontal_stack))

                # Draw gripper width
                gripper_width_im = plot_gripper_width(gripper_widths, gripper_detection_types, frame_i, width, height)[...,::-1]
                side_stack = np.vstack((gripper_width_im, clipped_depth_frame))

                final = np.hstack((vertical_stack, side_stack))

                # Write the frame
                out.write(final)

            # Release everything if job is finished
            ultrawide_cap.release()
            depth_cap.release()
            intermediate_cap.release()
            out.release()

            # Delete intermediate result
            os.remove(intermediate_video_out_path)

            print(f'{demonstration_to_display_string(demonstration_dir, side)} Wrote to {video_out_path}')
            num_processed += 1

        for side in get_demonstration_sides_present(demonstration_dir):
            visualize_side(side)

    print(f'\nVisualized {num_processed} demonstrations')
    print(f'Already visualized {num_already_processed} demonstrations')

def visualize_iphone_gopro_data(demonstration_iterator, cfg):
    num_processed = 0
    num_already_processed = 0
    for demonstration_dir in demonstration_iterator('demonstration'):
        def visualize_side(side):
            nonlocal num_processed, num_already_processed
            pose_path = os.path.join(demonstration_dir, f'camera_trajectory_gopro_{side}.csv')

            if not os.path.exists(pose_path):
                return
            
            df = pd.read_csv(pose_path)
            is_lost = df[['is_lost']].values[:, 0] # convert from (N, 1) to (N,)
            poses = df[['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']].values
            poses = pos_quat_xyzw_to_4x4(poses)
            
            video_out_path = f'{demonstration_dir}/combined_gopro_pose_and_video_{side}.mp4'
            if os.path.exists(video_out_path) and not cfg.overwrite:
                num_already_processed += 1
                return
            print(f'{demonstration_to_display_string(demonstration_dir, side)} Beginning to visualize on {side} side')
            
            # Load the pose data
            pd.read_csv(pose_path)
            
            # Visualize the frames and poses
            video_path = os.path.join(demonstration_dir, f'{side}.mp4')
            plot_video_aligned_trajectory(video_path, video_out_path, poses, is_lost, max_frames=cfg.visualize_til_frame)
            print(f'{demonstration_to_display_string(demonstration_dir, side)} Wrote to {video_out_path}')
            num_processed += 1

        visualize_side('left')
        visualize_side('right')

    print(f'\nVisualized {num_processed} demonstrations')
    print(f'Already visualized {num_already_processed} demonstrations')
