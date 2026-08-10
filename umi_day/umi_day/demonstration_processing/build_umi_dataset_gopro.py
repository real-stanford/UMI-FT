"""
Adapted from UMI codebase.
Replaces the main script for UMI SLAM pipeline to instead use the pose from iPhone.
python build_umi_dataset.py <session_dir>
"""

import sys
import os

# %%
import pathlib
import click
import subprocess

from umi_day.common.import_umi_source import get_umi_subprocess_env, get_umi_dir
from umi_day.common.replay_buffer_util import print_replay_buffer

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-o', '--use_original_dataset', is_flag=True, help='Use original UMI dataset format without additional label generation and task annotations')

def main(session_dir, use_original_dataset):
    umi_dir = get_umi_dir()
    session_dir = [pathlib.Path(os.path.expanduser(session)).absolute() for session in session_dir]
    modified_umi_orig_pipeline_scripts_dir = pathlib.Path('scripts_gopro_orig_pipeline')
    umi_tasks_pipeline_scripts_dir = pathlib.Path('scripts_gopro_tasks_pipeline')

    orig_umi_scripts_dir = pathlib.Path(os.path.join(umi_dir, 'scripts_slam_pipeline')).absolute()
    
    for session in session_dir:
        demo_dir = session.joinpath('demos')
        assert demo_dir.is_dir()

        session_name = os.path.basename(session)

        # we skip 00_process_videos as it is not needed for iPhone pose pipeline since processing happens with process_demos.py and create_session.py
        # we skip 01_extract_gopro_imu, 02_create_map, and 03_batch_slam as it is not needed for iPhone pose pipeline
        # we skip 04_detect_aruco because it is run in `process_demos.py` for each demonstration
        # we skip 05_run_calibrations because it is run in `process_demos.py` for each gripper calibration

        print("############# 06_generate_dataset_plan ###########")
        if use_original_dataset:
            script_path = modified_umi_orig_pipeline_scripts_dir.joinpath("06_generate_dataset_plan.py")
        else:
            script_path = umi_tasks_pipeline_scripts_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        if use_original_dataset:
            print("############# 07_generate_replay_buffer (UMI original version) ###########")
            script_path = orig_umi_scripts_dir.joinpath("07_generate_replay_buffer.py")
        else:
            print("############# 07_generate_replay_buffer (umi_day version) ###########")
            script_path = umi_tasks_pipeline_scripts_dir.joinpath("07_generate_replay_buffer.py")
        assert script_path.is_file()
        dataset_out_path = os.path.join(session, f'dataset_{session_name}.zarr')
        cmd = [
            'python', str(script_path),
            '-o', dataset_out_path,
            str(session)
        ]
        # notably we do not pass the --no_mirror flag here. In our gripper setup with iPhone we do not have mirrors because it would block more of the iPhones view for ARKit tracking and empirically mirrors do not provide much additional benefit. Passing --no_mirror masks out the mirror in the video, but we do not have a mirror so we do not need to mask it out (we want that space to be usable by the policy). This means that at deployment time you should also NOT pass the --no_mirror flag to the policy. The drawback of this is that you can't reuse your existing deployment hardware setups that do have mirrors (you will need to physically remove the mirrors before running policies trained on iPhone collected data).
        result = subprocess.run(cmd, env=get_umi_subprocess_env())
        assert result.returncode == 0

        print_replay_buffer(dataset_out_path)

## %%
if __name__ == "__main__":
    main()
