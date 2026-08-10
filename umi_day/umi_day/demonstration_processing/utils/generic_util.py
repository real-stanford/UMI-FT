import os
from datetime import datetime
import json
import yaml
from glob import glob
from omegaconf import DictConfig
import re
from umi_day.demonstration_processing.utils.color_util import red, green, yellow
from umi_day.common.timecode_util import datetime_fromisoformat
from pathlib import Path
import cv2
import numpy as np

def get_demonstration_video_frame_count(demonstration_dir, side):
    video_path = get_demonstration_main_video_path(demonstration_dir, side)
    assert os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_demonstration_video_fps(demonstration_dir, side):
    video_path = get_demonstration_main_video_path(demonstration_dir, side)
    assert os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_demonstration_json_data(dmeonstration_dir, side):
    json_path = os.path.join(dmeonstration_dir, f'{side}.json')
    assert os.path.exists(json_path)
    with open(json_path, 'r') as f:
        return json.load(f)

def get_demonstration_main_video_path(demonstration_dir, side):
    if is_iphone_only_demonstration(demonstration_dir):
        return os.path.join(demonstration_dir, f'{side}_rgb.mp4')
    else:
        return os.path.join(demonstration_dir, f'{side}.mp4')

def is_iphone_only_demonstration(demonstration_dir):
    side = get_demonstration_sides_present(demonstration_dir)[0]
    demonstration_json = get_demonstration_json_data(demonstration_dir, side)
    if 'hasGoPro' not in demonstration_json:
        return False
    else:
        return not demonstration_json['hasGoPro']

def get_demonstration_frame_times(demonstration_dir, side):
    if is_iphone_only_demonstration(demonstration_dir):
        demonstration_json = get_demonstration_json_data(demonstration_dir, side)
        return demonstration_json['rgbTimes']
    else: # GoPro
        metadata = read_demonstration_metadata(demonstration_dir)
        return metadata[f'{side}_gopro_frame_times']

def get_demonstration_sides_present(demonstration_dir):
    sides = []
    for side in ['left', 'right']:
        json_path = os.path.join(demonstration_dir, f'{side}.json')
        if os.path.exists(json_path):
            sides.append(side)
    return sides

def get_demonstration_property(demonstration_dir, side, property_name):
    demonstration_json = get_demonstration_json_data(demonstration_dir, side)
    return demonstration_json[property_name]

def get_gripper_calibration_run_dir(demonstration_dir, side):
    """Given a specific demonstration, returns the gripper calibration run directory. First look into the demonstration folder structure which has folders for each day, if if not present, then check in the same directory as the given demonstration"""
    # TODO: there are other places where this function can be used to simplify logic
    demonstration_json = get_demonstration_json_data(demonstration_dir, side)
    gripper_cal_run_name = demonstration_json['gripperCalibrationRunName']
    gripper_cal_run_dir = get_demonstration_path(get_demonstrations_dir_from_specific_dir(demonstration_dir), gripper_cal_run_name)

    # attempt search in same directory as given demonstration if previus search failed
    if not os.path.exists(gripper_cal_run_dir):
        gripper_cal_run_dir = os.path.join(os.path.dirname(demonstration_dir), gripper_cal_run_name.replace('_right', '').replace('_left', ''))
    
    assert os.path.exists(gripper_cal_run_dir)

    return gripper_cal_run_dir


def get_demonstrations_dir_from_specific_dir(demonstrations_dir):
    """Given a specific demonstration directory, return the parent demonstrations directory containing the rest of the demonstrations"""
    return Path(demonstrations_dir).parent.parent.as_posix()


def get_demonstration_path(demonstrations_dir, demonstration_name):
    demonstration_name = demonstration_name.replace('_right', '').replace('_left', '') # remove side from demonstration name
    demonstration_time_str_iso8601 = demonstration_name[:demonstration_name.index('T')]
    demonstration_time = datetime_fromisoformat(demonstration_time_str_iso8601)
    demonstration_ymd = demonstration_time.strftime('%Y-%m-%d')
    path = os.path.join(demonstrations_dir, demonstration_ymd, demonstration_name)
    return path


def demonstration_to_display_string(demonstration_dir, side=None):
    colored_demonstration_dir = green(demonstration_dir)
    if side is None:
        return f'[{colored_demonstration_dir}]'
    else:
        json_path = os.path.join(demonstration_dir, f'{side}.json')
        if not os.path.exists(json_path):
            return f'[{colored_demonstration_dir} {side}]'

        demonstration_json = get_demonstration_json_data(demonstration_dir, side)
        note = (f" \"{demonstration_json['note']}\"") if 'note' in demonstration_json and demonstration_json['note'] else ''
        return f'[{yellow(demonstration_json["sessionName"])} {colored_demonstration_dir} {side}{red(note)}]'

"""Demonstration metadata"""

def write_demonstration_metadata(demonstration_dir, metadata_dict, overwrite_all=False):
    if not overwrite_all:
        metadata_dict = {**read_demonstration_metadata(demonstration_dir), **metadata_dict}

    metadata_path = os.path.join(demonstration_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata_dict, f)


def read_demonstration_metadata(demonstration_dir):
    metadata_path = os.path.join(demonstration_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            yaml.safe_dump({}, f)
    
    with open(metadata_path, 'r') as f:
        metadata_dict = yaml.safe_load(f)
    return metadata_dict


def remove_key_demonstration_metadata(demonstration_dir, key):
    metadata = read_demonstration_metadata(demonstration_dir)
    if key in metadata:
        del metadata[key]
    write_demonstration_metadata(demonstration_dir, metadata, overwrite_all=True)

"""Filtering demonstrations"""

def get_demonstration_type(demonstration_dir):
    demonstration_name = os.path.basename(demonstration_dir)
    return demonstration_name.split('_')[-1]

def keep_demonstration(demonstration_title, demonstration_json_path, filters: DictConfig, demo_type=None, has_gopro=False):
    if type(demo_type) == str:
        demo_type = [demo_type]
    if demo_type is not None and not any([demonstration_title.endswith(f'_{type}') for type in demo_type]):
        return False

    if filters.demonstration_names is not None and demonstration_title not in filters.demonstration_names:
        return False
    
    if filters.demonstration_regex is not None and not re.search(filters.demonstration_regex, demonstration_title):
        return False
    
    if filters.task_names is not None:
        if 'gripper' not in demonstration_title:
            with open(demonstration_json_path, 'r') as f:
                    demonstration_json = json.load(f)
                    if 'taskNames' not in demonstration_json:
                        return False  
                    # if there is no intersection between the task names in the demo and the
                    # task names we are filtering for, we throw out the demo     
                    if not (bool(set(demonstration_json['taskNames']) & set(filters.task_names))):
                        return False
                
    if filters.session_name:
        split = demonstration_title.split('_')
        if len(split) == 4:
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type = split
            if filters.session_name != demonstration_session_name:
                return False
        else:
            # TODO: eventually remove this legacy format
            with open(demonstration_json_path, 'r') as f:
                demonstration_json = json.load(f)
                if 'sessionName' not in demonstration_json:
                    return False
                elif demonstration_json['sessionName'] != filters.session_name:
                    return False
                
    # Check if the demonstration has a GoPro video
    with open(demonstration_json_path, 'r') as f:
        demonstration_json = json.load(f)
        if 'hasGoPro' not in demonstration_json:
            # assume it's a GoPro demonstration
            if not has_gopro:
                return False
        elif demonstration_json['hasGoPro'] != has_gopro:
            return False
    
    return True


def iterate_demonstrations(demonstrations_dir, filters: DictConfig, demo_type=None, has_gopro=False):
    num_processed = 0
    for demonstration_dir in sorted(list(glob(demonstrations_dir + '/*/*'))):
        if not os.path.isdir(demonstration_dir):
            continue

        demonstration_json_path = os.path.join(demonstration_dir, f'left.json')
        if not os.path.exists(demonstration_json_path):
            demonstration_json_path = os.path.join(demonstration_dir, f'right.json')
        assert os.path.exists(demonstration_json_path)
        if not keep_demonstration(os.path.basename(demonstration_dir), demonstration_json_path, filters, demo_type, has_gopro=has_gopro):
            continue

        if num_processed == filters.max_demos:
            break

        yield demonstration_dir
        num_processed += 1


def demonstration_has_data_for_side(demonstration_dir, side):
    video_path = os.path.join(demonstration_dir, f'{side}.mp4')
    return os.path.exists(video_path)
