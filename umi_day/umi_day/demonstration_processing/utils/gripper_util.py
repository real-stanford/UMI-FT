import pickle
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from umi_day.demonstration_processing.utils.generic_util import get_gripper_calibration_run_dir, get_demonstration_video_frame_count, is_iphone_only_demonstration, get_demonstration_json_data
from umi_day.common.timecode_util import datetime_fromisoformat
from umi_day.common import import_umi_source
from umi.common.cv_util import get_gripper_width
from umi.common.interpolation_util import (
    get_gripper_calibration_interpolator, 
    get_interp1d
)
from umi_day.common.contants import IPHUMI_ULTRAWIDE_OFFSET_FROM_CENTER_X, IPHUMI_ULTRAWIDE_OFFSET_FROM_FINGER_AR_TAG_Z, IPHUMI_TCP_OFFSET_FROM_MAIN_CAMERA

def get_demo_gripper_width(demonstration_dir, side: str, include_detection_types: bool=False):
    if is_iphone_only_demonstration(demonstration_dir):
        return get_demo_gripper_width_iphone(demonstration_dir, side, include_detection_types)
    else:
        assert not include_detection_types
        return get_demo_gripper_width_gopro(demonstration_dir, side)


def get_demo_gripper_width_iphone(demonstration_dir, side: str, include_detection_types=False):
    """Returns a list of gripper widths for each frame in the demonstration. Adapted from 06_generate_dataset_plan.py from UMI.
    Note that since the iphone records ultrawide at 10Hz (and ultrawide is used to detect AR tags), but since we want to have gripper width for every frame of the main camera video (which is at 60Hz)

    Returns:
    - gripper_widths: list of gripper widths for each frame in the demonstration
    - gripper_detection_types: list of integers indicating the presence of the gripper in each frame. 0: not present, 1: both fingers present, 2: left finger only, 3: right finger only
    """

    # find the associated gripper calibration
    gripper_cal_run_dir = get_gripper_calibration_run_dir(demonstration_dir, side)

    # get the gripper calibration interpolator
    with open(Path(gripper_cal_run_dir).joinpath(f'{side}_gripper_range.json'), 'r') as f:
        gripper_range_data = json.load(f)
    max_width = gripper_range_data['max_width']
    min_width = gripper_range_data['min_width']
    gripper_cal_data = {
        'aruco_measured_width': [min_width, max_width],
        'aruco_actual_width': [min_width, max_width]
    }
    gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)

    # load the tag detection results for the demonstration
    pkl_path = Path(demonstration_dir).joinpath(f'{side}_tag_detection.pkl')
    with open(pkl_path, 'rb') as f:
        tag_detection_results = pickle.load(f)

    # identify the gripper id
    left_id = gripper_range_data['left_finger_tag_id']
    right_id = gripper_range_data['right_finger_tag_id']

    # extract the gripper width from the tag detection results
    detected_gripper_timestamps = list()
    detected_gripper_widths = list()
    detected_gripper_left_present = list()
    detected_gripper_right_present = list()
    for td in tag_detection_results:
        width, left_present, right_present = get_gripper_width_offset(td['tag_dict'], 
            left_id=left_id, right_id=right_id, 
            nominal_z=IPHUMI_ULTRAWIDE_OFFSET_FROM_FINGER_AR_TAG_Z, offset_x=IPHUMI_ULTRAWIDE_OFFSET_FROM_CENTER_X)
        if width is not None:
            detected_gripper_timestamps.append(td['time'])
            detected_gripper_widths.append(gripper_cal_interp(width))
            detected_gripper_left_present.append(left_present)
            detected_gripper_right_present.append(right_present)
    
    # some frames may not have had detections, so we interpolate to fill gaps
    gripper_interp = get_interp1d(detected_gripper_timestamps, detected_gripper_widths)

    # use the main RGB to get the timestamps. This is becuase the ultrawide records the tag detections at 10Hz, but we want to have gripper width for every frame of the main camera video (which is at 60Hz)
    demonstration_json = get_demonstration_json_data(demonstration_dir, side)
    video_times = demonstration_json['rgbTimes']
    video_times = [datetime_fromisoformat(t) for t in video_times]
    video_times = [(t - video_times[0]).total_seconds() for t in video_times]
    all_gripper_widths = gripper_interp(video_times)

    # compute the frame indices in the main camera video that correspond to the timestamps of gripper detections
    gripper_detection_types = np.zeros(len(video_times), dtype=np.int8)
    for detection_i in range(len(detected_gripper_timestamps)):
        detected_timestamp = detected_gripper_timestamps[detection_i]
        left_present = detected_gripper_left_present[detection_i]
        right_present = detected_gripper_right_present[detection_i]
        main_camera_frame_idx = np.argmin(np.abs(np.array(video_times) - detected_timestamp))
        if left_present and right_present:
            gripper_detection_types[main_camera_frame_idx] = 1
        elif left_present:
            gripper_detection_types[main_camera_frame_idx] = 2
        else:
            gripper_detection_types[main_camera_frame_idx] = 3

    if include_detection_types:
        return all_gripper_widths, gripper_detection_types
    else:
        return all_gripper_widths

def get_gripper_width_offset(tag_dict, left_id, right_id, nominal_z, offset_x, z_tolerance=0.08, only_side='left'):
    """Computes the width of the gripper from the tag detection results. If `only_side` is set to `left` or `right`, then the function will only rely on the tag detection on that side to perform the computation (if set to None then both sides are used)."""
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    use_left = only_side == 'left' or only_side is None
    use_right = only_side == 'right' or only_side is None

    left_x = None
    if left_id in tag_dict and use_left:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0] + offset_x

    right_x = None
    if right_id in tag_dict and use_right:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0] + offset_x

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    
    return width, left_x is not None, right_x is not None

def get_demo_gripper_width_gopro(demonstration_dir, side: str, nominal_z: float=0.072):
    """Returns a list of gripper widths for each frame in the demonstration. Adapted from 06_generate_dataset_plan.py from UMI."""

    # find the associated gripper calibration
    gripper_cal_run_dir = get_gripper_calibration_run_dir(demonstration_dir, side)

    # get the gripper calibration interpolator
    with open(Path(gripper_cal_run_dir).joinpath(f'{side}_gripper_range.json'), 'r') as f:
        gripper_range_data = json.load(f)
    max_width = gripper_range_data['max_width']
    min_width = gripper_range_data['min_width']
    gripper_cal_data = {
        'aruco_measured_width': [min_width, max_width],
        'aruco_actual_width': [min_width, max_width]
    }
    gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)

    # load the tag detection results for the demonstration
    pkl_path = Path(demonstration_dir).joinpath(f'{side}_tag_detection.pkl')
    with open(pkl_path, 'rb') as f:
        tag_detection_results = pickle.load(f)

    # identify the gripper id
    left_id = gripper_range_data['left_finger_tag_id']
    right_id = gripper_range_data['right_finger_tag_id']

    # extract the gripper width from the tag detection results
    all_gripper_timestamps = list()
    detected_gripper_timestamps = list()
    detected_gripper_widths = list()
    for td in tag_detection_results:
        width = get_gripper_width(td['tag_dict'], 
            left_id=left_id, right_id=right_id, 
            nominal_z=nominal_z)
        all_gripper_timestamps.append(td['time'])
        if width is not None:
            detected_gripper_timestamps.append(td['time'])
            detected_gripper_widths.append(gripper_cal_interp(width))
    
    # some frames may not have had detections, so we interpolate to fill gaps
    gripper_interp = get_interp1d(detected_gripper_timestamps, detected_gripper_widths)

    # it's possible that `tag_detection_results` actually contains 1 fewer frame then the demonstration video
    # this occurs becuase a discrepancy between the number of frames reported in the video and the actual number of frames that can be read out
    num_video_frames = get_demonstration_video_frame_count(demonstration_dir, side)
    if len(all_gripper_timestamps) == num_video_frames - 1:
        all_gripper_timestamps.append(all_gripper_timestamps[-1] + (all_gripper_timestamps[1] - all_gripper_timestamps[0]))

    all_gripper_widths = gripper_interp(all_gripper_timestamps)

    assert len(all_gripper_widths) == num_video_frames
    
    return all_gripper_widths

"""Transformations"""

def iphone_to_tcp_poses(iphone_poses):
    # the iPhone pose is the pose of the optical center of the main camera on the iPhone with respect to the world frame. With the iPhone in landscape mode (screen facing you), with the charging port to the right, the x-axis points to the right, the y-axis points up, and the z-axis points towards the user.
    # the tcp pose is the pose at the end of the gripper in the middle vertically and horizontally of the fingers. When holding the gripper in your hand, z points away from you, x points to the right, and y points down
    # thus to convert from iPhone pose to TCP pose, we need both a rotation and a translation
    # the rotation consists of a rotation to account for mount angle (15 degrees) and a rotation from arkit convention to TCP
    mount_rotation = np.array([[1,0,0],
                                [0,np.cos(np.deg2rad(15)),-np.sin(np.deg2rad(15))],
                                [0,np.sin(np.deg2rad(15)),np.cos(np.deg2rad(15))]])
    arkit_tcp_rot = np.array([[1,0,0],
                                [0,-1,0],
                                [0,0,-1]])
    iphone_tcp_rot = mount_rotation @ arkit_tcp_rot

    # iphone_poses is geometrically the pose of the iPhone frame in the world frame (W_T_I)

    # I_T_TCP is geometrically the pose of the TCP frame in the iPhone frame; involves a 180 rotation about X axis and a translation. For iPhone 15 Pro
    I_T_TCP = np.array([[0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,1]], dtype=np.float32)
    I_T_TCP[:3,:3] = iphone_tcp_rot
    I_T_TCP[:3,3] = IPHUMI_TCP_OFFSET_FROM_MAIN_CAMERA
    
    tcp_poses = np.array([W_T_I @ I_T_TCP for W_T_I in iphone_poses]) # now becomes W_T_TCP, which is geometrically the transform from the world frame to the TCP frame

    return tcp_poses
