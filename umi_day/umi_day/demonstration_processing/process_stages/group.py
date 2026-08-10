from omegaconf import DictConfig
import json
import os
from datetime import datetime
from av.error import InvalidDataError
from umi_day.common.generic_util import symlink_absolute
from glob import glob
import shutil
from umi_day.demonstration_processing.utils.color_util import red
from umi_day.demonstration_processing.utils.generic_util import write_demonstration_metadata, keep_demonstration, get_demonstration_path, demonstration_to_display_string
from umi_day.common.timecode_util import datetime_fromisoformat
from umi_day.demonstration_processing.utils.gopro_util import get_gopro_start_video_time

def group_iphone_data(cfg: DictConfig):
    """Given data from iPhone (potentially paired left right data or just one side), group them together into folders by demonstration."""
    new_demonstrations = set()
    existing_demonstrations = set()
    failure_demonstrations = set()
    demonstration_title_to_type = {}
    num_processed = 0

    # Load demonstrations from iPhone
    assert os.path.exists(cfg.iphone_dir), f'Did not find iphone directory: {cfg.iphone_dir}'
    demonstration_files = list(glob(os.path.join(cfg.iphone_dir, "**/*.json"), recursive=True))
    demonstration_files.sort(reverse=True)

    for demonstration_json_path in demonstration_files:
        demonstration_name = os.path.basename(demonstration_json_path).replace('.json', '')

        # get demonstration properties
        split = demonstration_name.split('_')
        if len(split) == 5:
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type, demonstration_side = split
            demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}_{demonstration_session_name}_{recording_type}'
        else:
            print(red(f"Skipping demonstration with unknown or old name format: {demonstration_name}"))
            continue

        # skip this demonstration under certain conditions
        if not keep_demonstration(demonstration_title, demonstration_json_path, cfg.filters):
            continue

        if num_processed == cfg.filters.max_demos:
            break
        num_processed += 1

        demonstration_in_dir = os.path.dirname(demonstration_json_path)
        demonstration_out_dir = get_demonstration_path(cfg.demonstrations_dir, demonstration_title)

        if os.path.exists(demonstration_out_dir) and cfg.overwrite and demonstration_title not in new_demonstrations:
            shutil.rmtree(demonstration_out_dir)

        demonstration_title_to_type[demonstration_title] = recording_type

        if not os.path.exists(demonstration_out_dir):
            new_demonstrations.add(demonstration_title)
            os.makedirs(demonstration_out_dir)
        elif demonstration_title not in new_demonstrations:
            existing_demonstrations.add(demonstration_title)
            continue

        # copy all the corresponding files to the output directory
        demonstration_in_files = glob(f'{demonstration_in_dir}/{demonstration_title}*')
        for in_file in demonstration_in_files:
            out_name = os.path.basename(in_file).replace(f'{demonstration_title}_', '') # looks like `right_depth.mp4` for example
            out_path = os.path.join(demonstration_out_dir, out_name)

            if cfg.symlink:
                symlink_absolute(in_file, out_path)
            else:
                shutil.copyfile(in_file, out_path)    

    def count_demonstrations_of_type(demonstrations_set, type):
        count = 0
        for demonstration in demonstrations_set:
            if type == demonstration_title_to_type[demonstration]:
                count += 1
        return count

    print(f'\nNew: {count_demonstrations_of_type(new_demonstrations, "demonstration")} demonstrations and {count_demonstrations_of_type(new_demonstrations, "grippercalibration")} gripper calibrations to {cfg.demonstrations_dir}')
    print(f'Existing: {count_demonstrations_of_type(existing_demonstrations, "demonstration")} demonstrations and {count_demonstrations_of_type(existing_demonstrations, "grippercalibration")} gripper calibrations')
    print(f'Failed: {count_demonstrations_of_type(failure_demonstrations, "demonstration")} demonstrations and {count_demonstrations_of_type(failure_demonstrations, "grippercalibration")} gripper calibrations. Failure names: {failure_demonstrations if len(failure_demonstrations) > 0 else "{}"}')


def group_iphone_gopro_data(cfg: DictConfig):
    """Given data from iPhone and GoPro cameras (potentially paired left right data or just one side), group them together by time (finding GoPro clips associated with iPhone recording)."""
    if cfg.skip_left:
        cfg.left_gopro_dir = None
    if cfg.skip_right:
        cfg.right_gopro_dir = None
    
    new_demonstrations = set()
    existing_demonstrations = set()
    failure_demonstrations = set()
    demonstration_title_to_type = {}
    num_processed = 0

    # Load demonstrations from iPhone
    assert os.path.exists(cfg.iphone_dir), f'Did not find iphone directory: {cfg.iphone_dir}'
    demonstration_files = list(glob(os.path.join(cfg.iphone_dir, "**/*.json"), recursive=True))
    demonstration_files.sort(reverse=True)

    for demonstration_json_path in demonstration_files:
        demonstration_name = os.path.basename(demonstration_json_path).replace('.json', '')

        # add this JSON file as a demonstration if it doesn't exist already
        split = demonstration_name.split('_')
        if len(split) == 5:
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type, demonstration_side = split
            demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}_{demonstration_session_name}_{recording_type}'
        elif len(split) == 4:
            # TODO: eventually remove this legacy format
            demonstration_time_str, demonstration_randomizer, demonstration_side, recording_type = split

            with open(demonstration_json_path, 'r') as f:
                json_data = json.load(f)
            demonstration_session_name = json_data['sessionName'] if 'sessionName' in json_data else 'no-session'
            demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}_{recording_type}'
        else:
            print(red(f"Skipping demonstration with unknown or old name format: {demonstration_name}"))
            continue

        # skip this demonstration under certain conditions
        if not keep_demonstration(demonstration_title, demonstration_json_path, cfg.filters, has_gopro=True):
            continue

        if num_processed == cfg.filters.max_demos:
            break
        num_processed += 1

        gopro_dir = cfg.left_gopro_dir if demonstration_side == 'left' else cfg.right_gopro_dir
        if gopro_dir is None:
            continue
        
        # start processing the demonstration
        demonstration_time_str_iso8601 = demonstration_time_str[:demonstration_time_str.index('T')] + demonstration_time_str[demonstration_time_str.index('T'):].replace('-', ':')
        demonstration_time = datetime_fromisoformat(demonstration_time_str_iso8601)

        demonstration_out_dir = get_demonstration_path(cfg.demonstrations_dir, demonstration_title)
        pose_out_path = os.path.join(demonstration_out_dir, demonstration_side + '.json')
        gopro_out_path = os.path.join(demonstration_out_dir, demonstration_side + '.mp4')

        if os.path.exists(demonstration_out_dir) and cfg.overwrite:
            shutil.rmtree(demonstration_out_dir)

        demonstration_title_to_type[demonstration_title] = recording_type

        if not os.path.exists(demonstration_out_dir):
            new_demonstrations.add(demonstration_title)
            os.makedirs(demonstration_out_dir)
        elif demonstration_title not in new_demonstrations:
            existing_demonstrations.add(demonstration_title)

        # check if outputs already exist
        if os.path.exists(pose_out_path) and os.path.exists(gopro_out_path):
            continue

        # find the corresponding GoPro video
        assert os.path.exists(gopro_dir), f'Did not find GoPro directory: {gopro_dir}'
        gopro_files = glob(os.path.join(gopro_dir, "**/*.MP4"), recursive=True)
        gopro_files.sort(reverse=True, key=lambda x: os.path.getmtime(x))

        assert len(gopro_files) > 0, f'No GoPro files found in {gopro_dir} for {demonstration_side} side'

        closest_gopro_file = ""
        closest_gopro_time_diff = float('inf')
        closest_gopro_time = datetime.min
        for gopro_file in gopro_files:
            try:
                gopro_timestamp = get_gopro_start_video_time(gopro_file) # get GoPro video start time using accurate timecode metadata stored in the mp4
            except IndexError:
                # some mp4 files are corrupted so just skip them
                continue
            except InvalidDataError:
                # some mp4 files are corrupted so just skip them
                continue

            time_diff = gopro_timestamp.timestamp() - demonstration_time.timestamp()

            if abs(time_diff) < abs(closest_gopro_time_diff):
                closest_gopro_time_diff = time_diff
                closest_gopro_file = gopro_file
                closest_gopro_time = gopro_timestamp
        
        print(f'{demonstration_to_display_string(demonstration_out_dir, demonstration_side)} Closest GoPro file was {closest_gopro_time_diff} seconds after the iPhone. GoPro ({demonstration_side}): {closest_gopro_time} ({os.path.basename(closest_gopro_file)}) and iPhone: {demonstration_time}')

        # TODO: we make the assumption that the GoPro video associated with the iPhone pose recording is the one that has the closest to the iPhone recording based on start time. Note that we use the GoPro start video time BEFORE calibrating with the QR code. This assumption is only reasonable if we assume the GoPro is only off from the QR code on the order of ~1 second
        if abs(closest_gopro_time_diff) > 2:
            print(red(f'--- ERROR: Previous demonstration ({demonstration_name}) had time error ({closest_gopro_time_diff}) seconds larger than 2 seconds between iPhone and GoPro. Skipping this demonstration ---'))
            shutil.rmtree(demonstration_out_dir)
            if demonstration_title in new_demonstrations:
                new_demonstrations.remove(demonstration_title)
            if demonstration_title in existing_demonstrations:
                existing_demonstrations.remove(demonstration_title)
            failure_demonstrations.add(demonstration_title)
            continue

        # Copy the GoPro video to the output directory if it doesn't already exist
        if not os.path.exists(gopro_out_path):
            if cfg.symlink:
                symlink_absolute(closest_gopro_file, gopro_out_path)
            else:
                shutil.copyfile(closest_gopro_file, gopro_out_path)

        # copy the pose data to the output directory if it doesn't exist
        if not os.path.exists(pose_out_path):
            if cfg.symlink:
                symlink_absolute(demonstration_json_path, pose_out_path)
            else:
                shutil.copyfile(demonstration_json_path, pose_out_path)

        # write metadata
        write_demonstration_metadata(demonstration_out_dir, {f'{demonstration_side}_closest_gopro_time_diff': closest_gopro_time_diff, f'{demonstration_side}_gopro_start_time_pre_qr_latancy_adjustment': closest_gopro_time.isoformat(), f'{demonstration_side}_demonstration_start_time': demonstration_time.isoformat()})

    def count_demonstrations_of_type(demonstrations_set, type):
        count = 0
        for demonstration in demonstrations_set:
            if type == demonstration_title_to_type[demonstration]:
                count += 1
        return count

    print(f'\nNew: {count_demonstrations_of_type(new_demonstrations, "demonstration")} demonstrations, {count_demonstrations_of_type(new_demonstrations, "qrcalibration")} QR calibrations, and {count_demonstrations_of_type(new_demonstrations, "grippercalibration")} gripper calibrations to {cfg.demonstrations_dir}')
    print(f'Existing: {count_demonstrations_of_type(existing_demonstrations, "demonstration")} demonstrations, {count_demonstrations_of_type(existing_demonstrations, "qrcalibration")} QR calibrations, and {count_demonstrations_of_type(existing_demonstrations, "grippercalibration")} gripper calibrations')
    print(f'Failed: {count_demonstrations_of_type(failure_demonstrations, "demonstration")} demonstrations, {count_demonstrations_of_type(failure_demonstrations, "qrcalibration")} QR calibrations, and {count_demonstrations_of_type(failure_demonstrations, "grippercalibration")} gripper calibrations. Failure names: {failure_demonstrations if len(failure_demonstrations) > 0 else "{}"}')
