"""Utilities for aligning and visualizing trajectories"""

import numpy as np
import open3d as o3d
from umi_day.common.latency_util import regular_sample, get_latency
from umi_day.common.transform_util import pose_4x4_to_6d, pose_6d_to_4x4, pose_4x4_to_quat_xyzw
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from datetime import datetime, timezone
from collections import OrderedDict
import pandas as pd
import os

def align_poses_different_timescales(pose1, pose2, time1, time2, show_alignment=False):
    # Use if pose timestamps are with respect to different timescales (their clocks are not aligned). This function aligns them using cross correlation. Returns aligned poses and times.

    # start both timescales at 0 (but at this point they are not aligned still)
    time1 -= min(time1)
    time2 -= min(time2)

    # Align trajectories using cross correlation
    t2 = time2
    x2 = pose_4x4_to_6d(pose2)
    t1 = time1
    x1 = pose_4x4_to_6d(pose1)

    n_dims = x1.shape[1]
    fig, axes = plt.subplots(n_dims, 3)
    fig.set_size_inches(15, 15, forward=True)

    # get independent latency for each dimension of position and rotation
    latencies = []
    for i in range(n_dims):
        latency, info = get_latency(x2[...,i], t2, x1[...,i], t1, force_positive=False)

        row = axes[i]
        ax = row[0]
        ax.plot(info['lags'], info['correlation'])
        ax.set_xlabel('lag')
        ax.set_ylabel('cross-correlation')
        ax.set_title(f"Action Dim {i} Cross Correlation")

        ax = row[1]
        ax.plot(t2, x2[...,i], label='target')
        ax.plot(t1, x1[...,i], label='actual')
        ax.set_xlabel('time')
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Raw observation")

        ax = row[2]
        t_samples = info['t_samples'] - info['t_samples'][0]
        ax.plot(t_samples, info['x_target'], label='target')
        ax.plot(t_samples-latency, info['x_actual'], label='actual-latency')
        ax.set_xlabel('time')
        
        ax.set_ylabel('gripper-width')
        ax.legend()
        ax.set_title(f"Action Dim {i} Aligned with latency={latency:.04f}")
        latencies.append(latency)

    fig.tight_layout()
    if show_alignment:
        plt.show()

    avg_latency = np.mean(sorted(latencies)[1:-1]) # cut outliers (assumed to be first and last entries)

    t1 -= avg_latency # shift actual time to align with target time

    pose1 = pose_6d_to_4x4(x1)
    pose2 = pose_6d_to_4x4(x2)
    
    return pose1, pose2, t1, t2

def error_between_poses(poses1, poses2):
    # compute position error
    pos_delta = poses1[:,:3,3] - poses2[:,:3,3] 
    pos_err = np.linalg.norm(pos_delta, axis=1).mean()

    # compute rotation error (pre cut)
    delta_rotation_mat = poses1[:, :3, :3] @ np.linalg.inv(poses2[:, :3, :3])
    delta_rotation_rotvec = Rotation.from_matrix(delta_rotation_mat).as_rotvec()
    rot_err = np.linalg.norm(delta_rotation_rotvec, axis=1).mean()

    return pos_err, rot_err

def world_align_poses(poses, world_align_transform=None):
    # normalize poses with respect to the specified world transform or, if None, such that the first pose is the identity transformation
    if world_align_transform is None:
        world_align_transform = np.linalg.inv(poses[0])
    poses = np.array([world_align_transform @ pose for pose in poses]) # normalize with respect to initial frame pose
    return poses, world_align_transform

def get_time_aligned_poses(pose1, pose2, time1, time2, resample_dt=1/60):
    # required that time1 and time2 are on the same time scale (meaning that the times are recorded with respect to the same reference time). It's not required that time1 and time2 represent the same time intervals. Returns 6d poses that are sampled at the same times. Poses are in 4x4 format.
    t_start = max(time1[0],time2[0])
    t_end = min(time1[-1],time2[-1])

    n_samples = int((t_end - t_start) / resample_dt)
    t_samples = np.arange(n_samples) * resample_dt + t_start

    pose1_samples = sample_poses_at_times(pose1, time1, t_samples)
    pose2_samples = sample_poses_at_times(pose2, time2, t_samples)

    return pose1_samples, pose2_samples, t_samples

def sample_poses_at_times(poses, pose_times, sample_times):
    poses = pose_4x4_to_6d(poses)
    pose_samples = np.array([regular_sample(poses[:,i], pose_times, sample_times) for i in range(6)]).T
    pose_samples = pose_6d_to_4x4(pose_samples)
    return pose_samples

def plot_time_aligned_trajectories(time_aligned_trajectories, colors=None, axis_size=0.04, include_base_frame=True):
    """TODO: update this function like the others to support headless rendering"""
    # time_aligned_trajectories is a list of matrices with shape (T, 4, 4), where T is the number of time steps. Returns list of Open3D geometry objects that can be visualized.
    T = len(time_aligned_trajectories[0])  # Number of time steps

    if colors is None:
        colors = [[1,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]  # Default colors

    assert len(colors) >= len(time_aligned_trajectories)

    geometries = []

    for i, trajectory in enumerate(time_aligned_trajectories):
        assert trajectory.shape == (T, 4, 4)

        trajectory_1 = trajectory[:,:3,3]

        # Create LineSet for trajectory 1
        lines_1 = [[i, i + 1] for i in range(T - 1)]
        colors_1 = [colors[i] for j in range(T - 1)]

        line_set_1 = o3d.geometry.LineSet()
        line_set_1.points = o3d.utility.Vector3dVector(trajectory_1)
        line_set_1.lines = o3d.utility.Vector2iVector(lines_1)
        line_set_1.colors = o3d.utility.Vector3dVector(colors_1)

        def plot_trajectory_with_rgb_axes(poses, K=1):
            """
            Plots the trajectory with RGB axes using Open3D.

            Parameters:
            - poses: numpy array of shape (T, 4, 4), where T is the number of trajectories.
            - K: Subsampling factor, use every K-th frame in the trajectory.
            """
            # Create a list to hold the frames for visualization
            frames = []
            
            # Subsample the poses
            subsampled_poses = poses[::K]
            
            for i, pose in enumerate(subsampled_poses):
                # Extract the translation vector
                translation = pose[:3, 3]
                
                # Extract the rotation matrix
                rotation = pose[:3, :3]
                
                # Create the coordinate frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                
                # Apply the rotation to the frame
                frame.rotate(rotation, center=translation)
                
                # Append the frame to the list
                frames.append(frame)
            
            # Create an Open3D visualization object
            return frames

        frames = plot_trajectory_with_rgb_axes(trajectory, K=30)

        if len(trajectory_1) > 1:
            geometries.append(line_set_1)
        if include_base_frame:
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2*axis_size)
            geometries.append(base_frame)
        
        # green sphere at the start of the trajectory
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
        sphere.translate(trajectory_1[0])
        sphere.paint_uniform_color([0, 1, 0])
        geometries.append(sphere)

        # red sphere at the end of the trajectory
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
        sphere.translate(trajectory_1[-1])
        sphere.paint_uniform_color([1, 0, 0])
        geometries.append(sphere)

        geometries.extend(frames)
    
    return geometries

def plot_time_aligned_trajectories_incremental(offscreen, scene_or_vis, time_aligned_trajectories, material, axis_size=0.04, include_base_frame=True, line_width=3, colors=None):
    """You can either provide an offscreen scene or a visualizer. This handles both properly"""
    T = len(time_aligned_trajectories[0])
    if colors is None:
        colors = [[0,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]

    line_material = o3d.visualization.rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = line_width

    def add_geo(name, geo, mat):
        if offscreen:
            scene = scene_or_vis
            scene.add_geometry(name, geo, mat)
        else:
            vis = scene_or_vis
            vis.add_geometry(geo)

    yield

    end_spheres = [None] * len(time_aligned_trajectories)
    for step_i in range(T-1):
        for traj_i, trajectory in enumerate(time_aligned_trajectories):            
            trajectory_1 = trajectory[step_i:step_i+2,:3,3]

            # add line
            points = o3d.utility.Vector3dVector(trajectory_1)
            lines = o3d.utility.Vector2iVector([[0, 1]])
            colors_1 = o3d.utility.Vector3dVector([colors[traj_i]])
            line_set = o3d.geometry.LineSet(points=points, lines=lines)
            line_set.colors = colors_1
            add_geo(f"line_{traj_i}_{step_i}", line_set, line_material)

            # Frame every N steps
            if step_i % 30 == 0:
                translation = trajectory[step_i][:3, 3]
                rotation = trajectory[step_i][:3, :3]
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                frame.rotate(rotation, center=translation)
                add_geo(f"frame_{traj_i}_{step_i}", frame, material)

            if step_i == 0 :
                # Green start sphere
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
                sphere.translate(trajectory_1[0])
                sphere.paint_uniform_color([0, 1, 0])
                add_geo(f"start_sphere_{traj_i}", sphere, material)

                if include_base_frame:
                    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2*axis_size)
                    add_geo("base_frame", base_frame, material)

            # Red moving end sphere
            if end_spheres[traj_i] is not None:
                if offscreen:
                    scene_or_vis.remove_geometry(f"end_sphere_{traj_i}")
                else:
                    scene_or_vis.remove_geometry(end_spheres[traj_i])
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=axis_size/4)
            sphere.translate(trajectory_1[-1])
            sphere.paint_uniform_color([1, 0, 0])
            add_geo(f"end_sphere_{traj_i}", sphere, material)
            end_spheres[traj_i] = sphere
        
        yield

def plot_video_aligned_trajectory(video_path, video_out_path, poses, is_lost=None, max_frames=-1, out_height=360, offscreen=True):
    """
    `offscreen` flag should be true if you are on a headless machine and want to use OffscreenRenderer and false if you are on a machine with a display and want to use Visualizer.
    Note that line width changes and the setting of up axis only apply if offscreen is true.
    """
    assert os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    assert T == len(poses)
    scale = out_height / H
    out_H = int(H * scale)
    out_W = int(W * scale)

    if is_lost is None:
        is_lost = np.zeros(T, dtype=bool)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    # Setup renderer
    if offscreen:
        renderer = o3d.visualization.rendering.OffscreenRenderer(out_W, out_H)
        scene = renderer.scene
        scene.set_background([1, 1, 1, 1])  # white background
    else:
        renderer = o3d.visualization.Visualizer()
        renderer.create_window(visible=True, width=out_W, height=out_H)
        scene = renderer

    video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc('a','v','c','1'), fps, (out_W*2, out_H))

    present_poses = poses[np.where(~is_lost)]
    geom_stepper = plot_time_aligned_trajectories_incremental(offscreen, scene, [present_poses], material, include_base_frame=False, colors=[[0,0,0]])
    
    num_present_poses = len(present_poses)
    max_frames = num_present_poses if max_frames == -1 else min(num_present_poses, max_frames)
    num_frames_visualized = 0

    with tqdm(total=max_frames, leave=False, desc='plot_video_aligned_trajectory') as pbar:
        for i in range(T):
            found_frame, video_frame = cap.read()
            assert found_frame

            if is_lost[i]:
                continue

            next(geom_stepper)

            if offscreen:
                # update camera position
                bbox = scene.bounding_box
                scene.camera.look_at(
                    center=bbox.get_center(),
                    eye=bbox.get_center() + 0.6 * bbox.get_extent() + np.array([0, 0, 0.1]),  # or scale based on bbox size
                    up=[0, 0, 1]
                )

                # Render to image
                rendered = renderer.render_to_image()
                rendered_np = np.asarray(rendered)
                rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
            else:
                # Update the visualizer with the current frame
                renderer.poll_events()
                renderer.update_renderer()

                # Capture the visualizer screen image
                trajectory_image = renderer.capture_screen_float_buffer(do_render=False)
                trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
                rendered_bgr = cv2.cvtColor(trajectory_image, cv2.COLOR_RGB2BGR)

            # Resize video frame
            video_frame = cv2.resize(video_frame, dsize=(out_W, out_H), interpolation=cv2.INTER_CUBIC)

            # Concatenate and write
            combined_image = np.hstack((video_frame, rendered_bgr))
            video_writer.write(combined_image)

            num_frames_visualized += 1
            pbar.update()
            if num_frames_visualized >= max_frames:
                break

    cap.release()
    video_writer.release()

    if not offscreen:
        cv2.destroyAllWindows()
        renderer.destroy_window()

def save_trajectory_umi_format(out_csv_path, poses, times, is_lost):
    # output the poses as csv (matching the format in UMI pipeline) which is in quat_xyzw and with relative timestamps starting at 0
    pos_quat_xyzw = pose_4x4_to_quat_xyzw(poses)

    csv_data = OrderedDict({
        'frame_idx': np.arange(len(times)),
        'timestamp': [time - times[0] for time in times],
        'state': [2] * len(times),
        'is_lost': is_lost,
        'is_keyframe': [False] * len(times),
        'x': pos_quat_xyzw[:, 0],
        'y': pos_quat_xyzw[:, 1],
        'z': pos_quat_xyzw[:, 2],
        'q_x': pos_quat_xyzw[:, 3],
        'q_y': pos_quat_xyzw[:, 4],
        'q_z': pos_quat_xyzw[:, 5],
        'q_w': pos_quat_xyzw[:, 6],
    })
    df = pd.DataFrame(csv_data)

    formats = {'timestamp': '{:.6f}'}
    for key in ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']:
        formats[key] = '{:.9f}'

    for col, f in formats.items():
        df[col] = df[col].map(lambda x: f.format(x))

    df.to_csv(out_csv_path, index=False)