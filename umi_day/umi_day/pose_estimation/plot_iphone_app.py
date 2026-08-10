"""Plots trajectory output from iPhone app"""

import json
import argparse
import os
import datetime
import open3d as o3d
import numpy as np

from umi_day.common.trajectory_util import get_time_aligned_poses, plot_time_aligned_trajectories

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_path', type=str, required=True, help='path to export excluding "_<side>.json"')
    args = parser.parse_args()

    # load data from whichever sides are available
    data_by_side = {}

    for side in ['left', 'right']:
        path = f'{args.export_path}_{side}.json'
        if not os.path.exists(path):
            continue

        with open(path, 'r') as f:
            side_data = json.load(f)

        times = side_data['poseTimes']
        times = [datetime.datetime.fromisoformat(time) for time in times]
        times = [time.timestamp() for time in times]

        poses = side_data['poseTransforms']
        poses = np.array(poses)

        data_by_side[side] = {
            'times': times,
            'poses': poses
        }

    assert 'left' in data_by_side or 'right' in data_by_side, 'Could not find files for left or right gripper'

    # if both left and right present, need to sample them at matching time intervals
    if 'left' in data_by_side and 'right' in data_by_side:
        # Both left and right sides are measured with respect to the same reference time (subject to inaccuracies between iPhone clocks), but won't be sampled at exactly the same times, so we need to resample them at matching times
        data_by_side['left']['poses'], data_by_side['right']['poses'], t_samples = get_time_aligned_poses(data_by_side['left']['poses'], data_by_side['right']['poses'], data_by_side['left']['times'], data_by_side['right']['times'])

        data_by_side['left']['times'] = t_samples
        data_by_side['right']['times'] = t_samples

    geometries = plot_time_aligned_trajectories([data_by_side[side]['poses'] for side in data_by_side])
    o3d.visualization.draw_geometries(geometries)
