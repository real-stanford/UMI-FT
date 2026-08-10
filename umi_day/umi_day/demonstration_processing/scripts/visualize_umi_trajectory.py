"""
Visualizes an UMI trajectory given a camera_trajectory.csv file and the GoPro video.
"""
from argparse import ArgumentParser
import os
from pandas import read_csv
from umi_day.common.trajectory_util import plot_video_aligned_trajectory, plot_time_aligned_trajectories
from umi_day.common.transform_util import pos_quat_xyzw_to_4x4
import json
import numpy as np
import open3d as o3d

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('trajectory_csv', type=str, help='Folder containing raw_video.mp4 and camera_trajectory.csv')
    parser.add_argument('--mode', choices=['video', 'viewer'], default='viewer', help='Whether to save the video or display it in a viewer')
    parser.add_argument('--viewer_size', default=1024, type=int, help='Size of the viewer window')
    parser.add_argument('--skip_tx_slam_tag', action='store_true', help='Skip loading tx_slam_tag.json to normalize trajectory with respect to AR tag during mapping')
    parser.add_argument('--trajectory_name', default='camera_trajectory.csv', type=str, help='Name of the trajectory file')
    parser.add_argument('--move_world_axis', default=True, type=bool, help='Move world axis position to match the first frame of the trajectory (rotation not changed)')
    args = parser.parse_args()

    assert args.trajectory_csv.endswith('.csv')

    input_dir = os.path.dirname(args.trajectory_csv)

    # Load the camera trajectory
    df = read_csv(args.trajectory_csv)
    is_lost = df[['is_lost']].values[:, 0]
    poses = df[['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']].values
    if args.move_world_axis:
        poses[:, :3] -= poses[0, :3]
    poses = pos_quat_xyzw_to_4x4(poses)

    # If mapping round exists in same directory level as input_dir, load it to normalize position with respect to the QR code
    tx_slam_tag_path = os.path.join(os.path.dirname(input_dir), 'mapping', 'tx_slam_tag.json')
    if os.path.exists(tx_slam_tag_path) and not args.skip_tx_slam_tag:
        print('Found mapping directory at same folder level as input_dir. Loading tx_slam_tag.json to normalize trajectory with respect to QR code during mapping.')
        with open(tx_slam_tag_path, 'r') as f:
            tx_slam_tag = np.array(json.load(f)['tx_slam_tag'])
        print(tx_slam_tag)
        tx_tag_slam = np.linalg.inv(tx_slam_tag)
        poses = np.array([tx_tag_slam @ pose for pose in poses])

    if args.mode == 'viewer':
        # Display the trajectory in a viewer
        poses = poses[np.where(~is_lost)] # don't draw poses that are lost
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=args.viewer_size, height=args.viewer_size)
        geometries = plot_time_aligned_trajectories([poses])
        for geometry in geometries:
            vis.add_geometry(geometry)
        while True:
            vis.poll_events()
            vis.update_renderer()
    elif args.mode == 'video':
        # Plot the trajectory
        if args.trajectory_csv.endswith('_left.csv'):
            side = '_left'
            video_path = os.path.join(input_dir, 'left.mp4')
        elif args.trajectory_csv.endswith('_right.csv'):
            side = '_right'
            video_path = os.path.join(input_dir, 'right.mp4')
        else:
            side = ''
            video_path = os.path.join(input_dir, 'raw_video.mp4')
        out_path = os.path.join(input_dir, f'combined_gopro_pose_and_video{side}.mp4')
        plot_video_aligned_trajectory(video_path, out_path, poses, is_lost)
        print(f'Saved video to {out_path}')
