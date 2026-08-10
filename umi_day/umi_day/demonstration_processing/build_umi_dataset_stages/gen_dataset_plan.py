import os
import pickle
import numpy as np
import json

from umi_day.demonstration_processing.utils.generic_util import get_demonstration_type, get_demonstration_sides_present, get_demonstration_json_data
from umi_day.demonstration_processing.utils.gripper_util import get_demo_gripper_width, iphone_to_tcp_poses
from umi_day.common.timecode_util import datetime_fromisoformat

from umi_day.common import import_umi_source
from umi.common.pose_util import mat_to_pose

def gen_dataset_plan(session_dir: str, out_plan_path: str, task_filters=None):
    # all_plans = [{
    #     "grippers": [{
    #         "tcp_pose": np.ndarray,
    #         "gripper_width": np.ndarray,
    #         "demo_start_pose": np.ndarray,
    #         "demo_end_pose": np.ndarray,
    #         "tag_corners": [np.ndarray]
    #     }],
    #     "cameras": [{
    #         "main_video_path": str,
    #         "depth_video_path": str,
    #         "ultrawide_video_path": str,
    #         "main_idx_to_ultrawide_idx": np.ndarray,
    #         "main_video_start_end": Tuple[int,int]
    #         "ultrawide_video_start_end": Tuple[int,int]
    #     }],
    #     "tasks": [
    #         {"name": str, "start_idx": int, "end_idx": int, "labels": {"label1": np.ndarray, ...}},
    #         ...,
    #     ]
    #     "episode_name": str
    # }]

    # add each demonstration to the plan
    all_plans = []
    demos_dir = os.path.join(session_dir, 'demos')
    for demo_name in os.listdir(demos_dir):
        demo_dir = os.path.join(demos_dir, demo_name)
        demonstration_type = get_demonstration_type(demo_dir)

        if demonstration_type == 'grippercalibration':
            continue
        else:
            assert demonstration_type == 'demonstration'

        sides_present = get_demonstration_sides_present(demo_dir)
        assert len(sides_present) == 1, 'Bimanual demonstrations not yet supported'
        side = sides_present[0] 

        # Load data
        json_data = get_demonstration_json_data(demo_dir, side)
        arkit_poses = np.array(json_data['poseTransforms'])
        T = arkit_poses.shape[0]
        
        """ gripper data """
        tcp_pose = mat_to_pose(iphone_to_tcp_poses(arkit_poses))
        gripper_width = get_demo_gripper_width(demo_dir, side)
        demo_start_pose = np.tile(tcp_pose[0], (T, 1))
        demo_end_pose = np.tile(tcp_pose[-1], (T, 1))

        gripper = {
            'tcp_pose': tcp_pose,
            'gripper_width': gripper_width,
            'demo_start_pose': demo_start_pose,
            'demo_end_pose': demo_end_pose,
        }

        """ camera data """
        main_video_path = os.path.join(demo_name, f'{side}_rgb.mp4')
        depth_video_path = os.path.join(demo_name, f'{side}_depth.mp4')
        ultrawide_video_path = os.path.join(demo_name, f'{side}_ultrawidergb.mp4')

        # since ultrawide images are recorded at 10Hz, we need to figure out which ultrawide image corresponds to each main image, which is recorded at 60Hz
        # to do this, we simply use the most recent ultrawide image captured. If there are frames before the first ultrawide image, we use the first ultrawide image available (which is technically in the future, but this is not a big problem).
        main_idx_to_ultrawide_idx = np.zeros(T, dtype=int)
        ultrawide_timestamps = np.array([datetime_fromisoformat(x).timestamp() if x != "" else 0 for x in json_data['ultrawideRGBTimes']])
        latest_ultrawide_frame_index = 0
        found_first_ultrawide = False
        for i in range(T):
            if ultrawide_timestamps[i] != 0:
                if found_first_ultrawide:
                    latest_ultrawide_frame_index += 1
                else:
                    found_first_ultrawide = True
            main_idx_to_ultrawide_idx[i] = latest_ultrawide_frame_index
        ultrawide_vid_length = latest_ultrawide_frame_index + 1
        camera = {
            'main_video_path': main_video_path,
            'depth_video_path': depth_video_path,
            'ultrawide_video_path': ultrawide_video_path,
            'main_idx_to_ultrawide_idx': main_idx_to_ultrawide_idx,
            'main_video_start_end': (0, T),
            'ultrawide_video_start_end': (0, ultrawide_vid_length)
        }
        assert ultrawide_vid_length == sum(ultrawide_timestamps != 0)

        """ task data """
        labels_path = os.path.join(demo_dir, 'labels.json')
        with open(labels_path, 'r') as f:
            tasks = json.load(f)
        
        if task_filters:
            task_set = set(task_filters)
            tasks = [t for t in tasks if t["name"] in task_set]

        # put all data into plan
        plan = {
            'grippers': [gripper],
            'cameras': [camera],
            'tasks': tasks,
            'episode_name': demo_name
        }
        all_plans.append(plan)
    
    # save plan
    with open(out_plan_path, 'wb') as f:
        pickle.dump(all_plans, f)
    print(f'Wrote dataset plan to {out_plan_path}')
