"""Given a folder of processed demonstrations, generates a session folder containing the demonstrations you want to use to train a policy."""

import os
import shutil
import re
from glob import glob
import hydra
from omegaconf import DictConfig

from umi_day.common.generic_util import symlink_absolute
from umi_day.demonstration_processing.utils.generic_util import get_demonstration_sides_present, get_demonstration_json_data
from utils.generic_util import get_demonstration_path

@hydra.main(version_base="1.2", config_path="config", config_name="create_session_iphone")
def main(cfg: DictConfig):
    # Make sure demonstration dir exists
    demonstrations_dir = cfg.demonstrations_dir
    assert os.path.isdir(demonstrations_dir)

    # Create a session folder
    session_dir = os.path.join(cfg.sessions_dir, cfg.output_session_name)
    if os.path.exists(session_dir):
        if cfg.overwrite:
            print(f'Overwriting existing session at {session_dir}')
            shutil.rmtree(session_dir)
        else:
            print(f'Session already exists at {session_dir}')
            exit()
    os.makedirs(session_dir, exist_ok=True)

    # demos
    demos_dir = os.path.join(session_dir, 'demos')
    os.makedirs(demos_dir, exist_ok=True)

    # process demonstrations
    gripper_calibrations_dirs = set()
    num_demonstrations, num_gripper_calibrations = 0, 0
    def process_demonstration_dir(demonstration_dir, include_gripper_calibrations=False):
        nonlocal num_demonstrations, num_gripper_calibrations
        if os.path.isdir(demonstration_dir):
            base_demo_dir_name = os.path.basename(demonstration_dir)
            session_demo_dir = os.path.join(demos_dir, base_demo_dir_name)
            if os.path.exists(session_demo_dir):
                return

            is_demonstration = base_demo_dir_name.endswith('_demonstration')
            is_gripper_calibration = base_demo_dir_name.endswith('_grippercalibration')
            
            if is_demonstration:
                if num_demonstrations >= cfg.max_demos and cfg.max_demos >= 0:
                    return

                num_demonstrations += 1

                # make sure we add the gripper calibration to the session
                for side_present in get_demonstration_sides_present(demonstration_dir):
                    json_data = get_demonstration_json_data(demonstration_dir, side_present)
                    gripper_calibrations_dirs.add(get_demonstration_path(demonstrations_dir, json_data['gripperCalibrationRunName']))
            elif is_gripper_calibration:
                if not include_gripper_calibrations:
                    return

                num_gripper_calibrations += 1
            else:
                raise NotImplementedError

            # Symlink the demonstration into the session
            symlink_absolute(demonstration_dir, session_demo_dir, target_is_directory=True)

            print(f'Added {base_demo_dir_name} to session')

    # Copy the processed demonstrations by name filter
    for filter in cfg.input_name_filters:
        for demonstration_dir in glob(demonstrations_dir + "/*/" + filter): 
            process_demonstration_dir(demonstration_dir)

    # Copy the processed demonstrations by session name filter
    for filter in cfg.input_session_filters:
        for demonstration_dir in glob(demonstrations_dir + "/*/*"):
            demonstration_name = os.path.basename(demonstration_dir)
            split = demonstration_name.split('_')
            if len(split) != 4:
                continue
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type = split
                
            if re.match(filter, demonstration_session_name):
                process_demonstration_dir(demonstration_dir)

    # Copy all the associated gripper calibrations (it's possible that the gripper calibration is under a different session name or demonstration title filter that doesn't match the specified filters) so we want to manually include them
    for demonstration_dir in gripper_calibrations_dirs:
        process_demonstration_dir(demonstration_dir, include_gripper_calibrations=True)

    print(f'Finished creating session at {session_dir} with {num_demonstrations} demonstrations and {num_gripper_calibrations} gripper calibrations')

if __name__ == '__main__':
    main()
