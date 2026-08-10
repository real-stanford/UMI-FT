"""
Adapted from UMI codebase.
Replaces the main script for UMI SLAM pipeline to instead use the pose from iPhone.
python build_umi_dataset.py <session_dir>
"""

import os
import hydra
from omegaconf import DictConfig

from umi_day.demonstration_processing.build_umi_dataset_stages.gen_dataset_plan import gen_dataset_plan
from umi_day.demonstration_processing.build_umi_dataset_stages.gen_replay_buffer import gen_replay_buffer
from umi_day.common.replay_buffer_util import print_replay_buffer

@hydra.main(config_path="config", config_name="build_umi_dataset_iphone")
def main(cfg: DictConfig):
    session_dir = cfg.session_dir
    overwrite = cfg.overwrite
    stages = cfg.stages
    assert all(x in ['dataset_plan', 'replay_buffer'] for x in stages)

    session_name = os.path.basename(session_dir)
    demo_dir = os.path.join(session_dir, 'demos')
    assert os.path.isdir(demo_dir)

    dataset_plan_path = os.path.join(session_dir, 'dataset_plan.pkl')

    if 'dataset_plan' in stages:
        if os.path.exists(dataset_plan_path):
            if overwrite:
                os.remove(dataset_plan_path)
            else:
                print(f'Dataset plan already exists at {dataset_plan_path} and overwrite is not set')
                exit()
        gen_dataset_plan(session_dir, dataset_plan_path, cfg.task_names)

    if 'replay_buffer' in stages:
        replay_buffer_path = os.path.join(session_dir, f'replay_buffer_{session_name}.zarr.zip')
        if os.path.exists(replay_buffer_path):
            if overwrite:
                os.remove(replay_buffer_path)
            else:
                print(f'Replay buffer already exists at {replay_buffer_path} and overwrite is not set')
                exit()
        gen_replay_buffer(session_dir, dataset_plan_path, replay_buffer_path, cfg.replay_buffer)
        print_replay_buffer(replay_buffer_path, vis_iphone_rgb=cfg.replay_buffer.vis_rgb, vis_iphone_video=cfg.replay_buffer.vis_video)

if __name__ == "__main__":
    main()
