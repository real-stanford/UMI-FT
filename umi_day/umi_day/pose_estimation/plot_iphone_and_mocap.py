"""Takes in an iPhone and MoCap trajectory (not necessarily time aligned) and aligns them. Includes quantitative metrics for position and rotation error of iPhone relative to mocap and includes trajectory visualization."""

import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from umi_day.common.trajectory_util import get_time_aligned_poses, plot_time_aligned_trajectories, world_align_poses, error_between_poses, align_poses_different_timescales
from umi_day.common.transform_util import pose_7d_to_4x4

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--show_alignment', action='store_true')
    args = parser.parse_args()

    multi_path = args.path
    trajectory = np.load(open(multi_path,'rb'),allow_pickle=True).item()

    geometries = []

    iphone_world_align_transform, mocap_world_align_transform = None, None

    for side in ['left', 'right']:
        if side in trajectory['iphone']:
            iphone_pose = pose_7d_to_4x4(trajectory['iphone'][side]['pose'])
            mocap_pose = pose_7d_to_4x4(trajectory['mocap'][side]['pose'])

            iphone_time = trajectory['iphone'][side]['time']
            mocap_time = trajectory['mocap'][side]['time']

            # we want both left and right trajectories of iphone to be normalized with respect to the same world frame (same for mocap)
            iphone_pose, iphone_world_align_transform = world_align_poses(iphone_pose, iphone_world_align_transform)
            mocap_pose, mocap_world_align_transform = world_align_poses(mocap_pose, mocap_world_align_transform)

            # initial error between iphone and mocap
            pos_err_pre_cut, rot_err_pre_cut = error_between_poses(iphone_pose,mocap_pose)

            # iphone and mocap are on different time scales, so we need to align them via cross correlation
            iphone_pose, mocap_pose, iphone_t, mocap_t = align_poses_different_timescales(iphone_pose, mocap_pose, iphone_time, mocap_time, show_alignment=args.show_alignment)

            # time alignment correction is done, now resample at matching points in the overlapping time interval
            iphone_pose, mocap_pose, aligned_sample_times = get_time_aligned_poses(iphone_pose, mocap_pose, iphone_t, mocap_t)

            # compute pose error with aligned trajectories
            pos_err_post_cut, rot_err_post_cut = error_between_poses(iphone_pose,mocap_pose)

            print(f'Average position error: {pos_err_post_cut*100} centimeters (was {pos_err_pre_cut*100} centimeters pre cut)')
            print(f'Average rotation error: {rot_err_post_cut} radians (was {rot_err_pre_cut} radians pre cut)')

            # plot the trajectories
            cur_geoms = plot_time_aligned_trajectories([iphone_pose, mocap_pose])

            geometries.extend(cur_geoms)
    
    assert geometries, 'Neither left or right poses were found in the trajectory'

    o3d.visualization.draw_geometries(geometries)
