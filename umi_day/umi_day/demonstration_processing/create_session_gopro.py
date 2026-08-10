"""Given a folder of processed demonstrations, generates a session folder containing the demonstrations you want to use to train a policy."""

import argparse
from glob import glob
import os
import shutil
import json
import re

from umi_day.common.generic_util import symlink_absolute, symlink_relative
from utils.generic_util import get_demonstration_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demonstrations_dir', type=str, default='tmp_demonstrations', help='path to the folder containing processed demonstrations')
    parser.add_argument('--sessions_dir', type=str, default='tmp_sessions', help='path to the folder to save the session')
    parser.add_argument('--input_name_filters', type=str, nargs='*', default=[], help='glob filters on demonstration names to keep. If you have all your demonstrations under a session name, then you should use --input_session_filters instead of this argument') # TODO: this is inconsistent with the `--demonstrations_filter` argument in process_demos.py that uses regex filters instead of glob filters. See if we can make this consistent
    parser.add_argument('--input_session_filters', type=str, nargs='*', default=[], help='regex filters on demonstration session names to keep')
    parser.add_argument('--output_session_name', type=str, required=True, help='name of the output session to create')
    parser.add_argument('--max_demos', type=int, default=-1, help='maximum number of demonstrations to include in the session')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing session if exists')
    args = parser.parse_args()

    # Make sure demonstration dir exists
    demonstrations_dir = args.demonstrations_dir
    assert os.path.isdir(demonstrations_dir)

    # Create a session folder
    session_dir = os.path.join(args.sessions_dir, args.output_session_name)
    if os.path.exists(session_dir):
        if args.overwrite:
            print(f'Overwriting existing session at {session_dir}')
            shutil.rmtree(session_dir)
        else:
            print(f'Session already exists at {session_dir}')
            exit()
    os.makedirs(session_dir, exist_ok=True)

    # demos
    demos_dir = os.path.join(session_dir, 'demos')
    os.makedirs(demos_dir, exist_ok=True)

    # raw_videos
    raw_videos_dir = os.path.join(session_dir, 'raw_videos')
    os.makedirs(raw_videos_dir, exist_ok=True)

    # process demonstrations
    gripper_calibrations_dirs = set()
    num_demonstrations, num_gripper_calibrations = 0, 0
    def process_demonstration_dir(demonstration_dir, include_gripper_calibrations=False):
        global num_demonstrations, num_gripper_calibrations
        if os.path.isdir(demonstration_dir):
            base_demo_dir_name = os.path.basename(demonstration_dir)
            is_demonstration = base_demo_dir_name.endswith('_demonstration')
            is_gripper_calibration = base_demo_dir_name.endswith('_grippercalibration')
            if not (is_demonstration or is_gripper_calibration):
                return
            
            if is_demonstration:
                if num_demonstrations >= args.max_demos and args.max_demos >= 0:
                    return

                left_gopro_trjajectory_csv = os.path.join(demonstration_dir, 'camera_trajectory_gopro_left.csv')
                right_gopro_trjajectory_csv = os.path.join(demonstration_dir, 'camera_trajectory_gopro_right.csv')

                left_present = os.path.exists(left_gopro_trjajectory_csv)
                right_present = os.path.exists(right_gopro_trjajectory_csv)

                if left_present and right_present:
                    # bimuanual case
                    print(f'--- WARNING: Skipping {base_demo_dir_name} as bimanual case not implemented yet ---')
                    return
                elif left_present:
                    side_present = 'left'
                elif right_present:
                    side_present = 'right'
                else:
                    print(f'--- WARNING: Skipping {base_demo_dir_name} as no processed video found for either left or right ---')
                    return

                demo_name = f'demo_{base_demo_dir_name}'
                destination_raw_video_path = os.path.join(raw_videos_dir, f'{base_demo_dir_name}.mp4')
                num_demonstrations += 1

                # make sure we add the gripper calibration to the session
                if left_present:
                    with open(os.path.join(demonstration_dir, 'left.json'), 'r') as f:
                        data = json.load(f)
                    gripper_calibrations_dirs.add(get_demonstration_path(demonstrations_dir, data['gripperCalibrationRunName']))
                if right_present:
                    with open(os.path.join(demonstration_dir, 'right.json'), 'r') as f:
                        data = json.load(f)
                    gripper_calibrations_dirs.add(get_demonstration_path(demonstrations_dir, data['gripperCalibrationRunName']))

            elif is_gripper_calibration:
                if not include_gripper_calibrations:
                    return
                left_path = os.path.join(demonstration_dir, 'left.mp4')
                right_path = os.path.join(demonstration_dir, 'right.mp4')

                if os.path.exists(left_path) and os.path.exists(right_path):
                    # bimuanual case
                    print(f'--- WARNING: Skipping {base_demo_dir_name} as bimanual case not implemented yet ---')
                    return
                elif os.path.exists(left_path):
                    side_present = 'left'
                elif os.path.exists(right_path):
                    side_present = 'right'
                else:
                    print(f'--- WARNING: Skipping {base_demo_dir_name} as no video found for either left or right ---')
                    return

                demo_name = f'gripper_calibration_{base_demo_dir_name}'
                raw_videos_gripper_calibration_dir = os.path.join(raw_videos_dir, f'gripper_calibration_{num_gripper_calibrations}')
                os.makedirs(raw_videos_gripper_calibration_dir)
                destination_raw_video_path = os.path.join(raw_videos_gripper_calibration_dir, f'{base_demo_dir_name}.mp4')

                num_gripper_calibrations += 1
            else:
                raise NotImplementedError

            # Create the demonstration dir in the session dir
            session_demo_dir = os.path.join(demos_dir, demo_name)
            os.makedirs(session_demo_dir, exist_ok=True)
            
            # Symlink camera trajectory
            if is_demonstration:
                symlink_absolute(os.path.join(demonstration_dir, f'camera_trajectory_gopro_{side_present}.csv'), os.path.join(session_demo_dir, 'camera_trajectory.csv'))

            # Symlink the AR tag detection
            symlink_absolute(os.path.join(demonstration_dir, f'tag_detection_{side_present}.pkl'), os.path.join(session_demo_dir, 'tag_detection.pkl'))

            # Symlink the gripper range calibration
            if is_gripper_calibration:
                symlink_absolute(os.path.join(demonstration_dir, f'gripper_range_{side_present}.json'), os.path.join(session_demo_dir, 'gripper_range.json'))

            # Symlink the raw video to the demo dir
            original_video_path = os.path.join(demonstration_dir, f'{side_present}.mp4')
            demo_dir_raw_video_path = os.path.join(session_demo_dir, 'raw_video.mp4')
            symlink_absolute(original_video_path, demo_dir_raw_video_path) # raw video is a somewhat misleading name since it's actually the trimmed post-processed video after aligning with iPhone, but we use the name `raw_video.mp4` to match the UMI naming convention

            # Symlink the labels (if it exists)
            labels_path = os.path.join(demonstration_dir, f'labels.json')
            if os.path.exists(labels_path):
                symlink_absolute(labels_path, os.path.join(session_demo_dir, 'labels.json'))

            # Symlink the original demonstration dir into the session demo dir
            symlink_absolute(demonstration_dir, os.path.join(session_demo_dir, 'original_demo'), target_is_directory=True)
    
            # Symlink the raw video to the raw_videos folder
            symlink_relative(demo_dir_raw_video_path, destination_raw_video_path)

            print(f'Added {base_demo_dir_name} to session')

    # Copy the processed demonstrations by name filter
    for filter in args.input_name_filters:
        for demonstration_dir in glob(demonstrations_dir + "/*/" + filter): 
            process_demonstration_dir(demonstration_dir)

    # Copy the processed demonstrations by session name filter
    for filter in args.input_session_filters:
        # TODO: eventually session name will be put in the title of the demonstration in which case we can just look at the demontsration name rather than having to open the json file
        for demonstration_dir in glob(demonstrations_dir + "/*/*"):
            demonstration_name = os.path.basename(demonstration_dir)
            split = demonstration_name.split('_')
            if len(split) == 4:
                demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type = split
            else:
                # TODO: legacy support for old demonstration naming format. Remove this
                left_json_path = os.path.join(demonstration_dir, 'left.json')
                right_json_path = os.path.join(demonstration_dir, 'right.json')
                json_path = right_json_path if os.path.exists(right_json_path) else left_json_path

                with open(json_path, 'r') as f:
                    data = json.load(f)

                demonstration_session_name = data['sessionName'] if 'sessionName' in data else "no-session"
                
            if re.match(filter, demonstration_session_name):
                process_demonstration_dir(demonstration_dir)

    # Copy all the associated gripper calibrations (it's possible that the gripper calibration is under a different session name or demonstration title filter that doesn't match the specified filters) so we want to manually include them
    for demonstration_dir in gripper_calibrations_dirs:
        process_demonstration_dir(demonstration_dir, include_gripper_calibrations=True)

    # Create a mapping folder
    mapping_dir = os.path.join(demos_dir, 'mapping')
    os.makedirs(mapping_dir)
    tx_slam_tag_path = os.path.join(mapping_dir, 'tx_slam_tag.json')
    with open(tx_slam_tag_path, 'w') as f:
        json.dump({
            'tx_slam_tag': [
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]
            ]
        }, f)

    print(f'Finished creating session at {session_dir} with {num_demonstrations} demonstrations and {num_gripper_calibrations} gripper calibrations')
