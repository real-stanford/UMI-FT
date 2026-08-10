from omegaconf import DictConfig
import json
import os
from datetime import datetime, timezone
import cv2
from umi_day.common.trajectory_util import sample_poses_at_times, save_trajectory_umi_format
from umi_day.common.transform_util import pose_4x4_to_quat_xyzw, pos_quat_xyzw_to_4x4
import numpy as np
import math
from umi_day.demonstration_processing.utils.generic_util import get_demonstration_path, write_demonstration_metadata
from umi_day.common.timecode_util import datetime_fromisoformat
from umi_day.demonstration_processing.utils.color_util import red
from umi_day.demonstration_processing.utils.gopro_util import get_gopro_start_video_time

def align_iphone_gopro_data(demonstration_iterator, cfg: DictConfig):
    num_processed = 0
    num_already_processed = 0
    for demonstration_dir in demonstration_iterator('demonstration'):
        # align the pose data with the video
        left_pose_path = os.path.join(demonstration_dir, 'left.json')
        right_pose_path = os.path.join(demonstration_dir, 'right.json')
        left_video_path = os.path.join(demonstration_dir, 'left.mp4')
        right_video_path = os.path.join(demonstration_dir, 'right.mp4')
        left_present = os.path.exists(left_pose_path) and os.path.exists(left_video_path)
        right_present = os.path.exists(right_pose_path) and os.path.exists(right_video_path)

        # check if we have already performed the alignment
        finished = True
        def check_finished(side):
            nonlocal finished
            camera_trajectory_path = os.path.join(demonstration_dir, f'camera_trajectory_gopro_{side}.csv')
            finished = finished and os.path.exists(camera_trajectory_path)

        if left_present:
            check_finished('left')
        if right_present:
            check_finished('right')
        
        if finished and not cfg.overwrite:
            num_already_processed += 1
            continue
        else:
            print(f'[{demonstration_dir}] Aligning...')
            num_processed += 1
        
        # load times from pose data
        def load_pose_data(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            return pose_data
            
        def load_pose_times(pose_data):
            times = pose_data['poseTimes']
            times = [datetime_fromisoformat(x).timestamp() for x in times]
            assert len(times) == len(pose_data['poseTransforms'])

            return times
        
        left_pose_data = load_pose_data(left_pose_path) if left_present else None
        right_pose_data = load_pose_data(right_pose_path) if right_present else None

        left_pose_times = load_pose_times(left_pose_data) if left_present else []
        right_pose_times = load_pose_times(right_pose_data) if right_present else []

        left_pose_start_time = left_pose_times[0] if left_present else -1
        right_pose_start_time = right_pose_times[0] if right_present else -1
        left_pose_end_time = left_pose_times[-1] if left_present else -1
        right_pose_end_time = right_pose_times[-1] if right_present else -1

        # load times from video metadata
        def load_gopro_video_times(video_path, side):
            """Only works on GoPro mp4 files which have special metadata for start time"""
            # Load the QR latency relative to the GoPro video
            pose_data = left_pose_data if side == 'left' else right_pose_data
            calibration_demonstration_name = pose_data['qrCalibrationRunName']
            demonstration_dir = get_demonstration_path(cfg.demonstrations_dir, calibration_demonstration_name)
            qr_calibration_path = os.path.join(demonstration_dir, f'processed_{side}.json')
            with open(qr_calibration_path, 'r') as f:
                qr_calibration_data = json.load(f)
                ms_gopro_ahead_of_qr = qr_calibration_data["msGoProAheadOfQR"]

            start_time = get_gopro_start_video_time(video_path, gopro_latency_correction_ms=-ms_gopro_ahead_of_qr).timestamp()

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()

            end_time = start_time + duration
            return start_time, end_time, fps, frame_count
        
        left_video_start_time, left_video_end_time, _, left_frame_count = load_gopro_video_times(left_video_path, 'left') if left_present else (-1, -1, -1, -1)
        right_video_start_time, right_video_end_time, _, right_frame_count = load_gopro_video_times(right_video_path, 'right') if right_present else (-1, -1, -1, -1)

        if (left_frame_count != -1 and left_frame_count < cfg.min_frame_count) or (right_frame_count != -1 and right_frame_count < cfg.min_frame_count):
            print(red(f'--- ERROR: Skipping {demonstration_dir} because one of the GoPro videos has less than {cfg.min_frame_count} frames ---'))
            continue

        # determine the start and end pose by finding where all the data overlaps
        valid_start_times = []
        valid_end_times = []
        if left_pose_times:
            valid_start_times.extend([left_pose_start_time, left_video_start_time])
            valid_end_times.extend([left_pose_end_time, left_video_end_time])
        if right_pose_times:
            valid_start_times.extend([right_pose_start_time, right_video_start_time])
            valid_end_times.extend([right_pose_end_time, right_video_end_time])
        
        valid_start_time = max(valid_start_times)
        valid_end_time = min(valid_end_times)

        def sample_poses_at_video_frames(video_path, side, poses, pose_times, start_time, end_time):
            video_start_time, video_end_time, fps, frame_count = load_gopro_video_times(video_path, side)
            
            # cut frames that happen before start_time
            time_cut_from_start = start_time - video_start_time
            start_frames_to_cut_float = time_cut_from_start * fps
            start_frames_to_cut = math.ceil(start_frames_to_cut_float)
            assert start_frames_to_cut >= 0
            extra_time_cut_from_start = (start_frames_to_cut - start_frames_to_cut_float) / fps # this is the tiny bit of time (less than ~1/60 of a second) that is cut before the first video frame we use

            # cut frames that happen after end_time
            video_duration = video_end_time - video_start_time - time_cut_from_start - extra_time_cut_from_start
            valid_duration = end_time - start_time
            end_frames_to_cut = math.ceil((video_duration + extra_time_cut_from_start - valid_duration) * fps)
            assert end_frames_to_cut >= 0

            # sample poses at the same times as the video frames
            resample_dt = 1 / fps
            num_frames_to_keep = frame_count - start_frames_to_cut - end_frames_to_cut
            sample_times = np.arange(num_frames_to_keep) * resample_dt + start_time + extra_time_cut_from_start
            assert sample_times[0] >= pose_times[0] and sample_times[-1] <= pose_times[-1]
            pose_samples = sample_poses_at_times(poses, pose_times, sample_times)

            # save some metadata
            write_demonstration_metadata(demonstration_dir, {f'{side}_time_cut_from_start': time_cut_from_start, f'{side}_extra_time_cut_from_start': extra_time_cut_from_start, f'{side}_start_frames_to_cut': start_frames_to_cut, f'{side}_end_frames_to_cut': end_frames_to_cut, f'{side}_num_frames_to_keep': num_frames_to_keep, f'{side}_raw_frame_count': frame_count})
            
            return num_frames_to_keep, pose_samples, sample_times, start_frames_to_cut, end_frames_to_cut
        
        # save poses and frames
        def save_sampled(num_frames_to_keep, iphone_cam_poses, times, original_pose_data, start_frames_to_cut, end_frames_to_cut):
            side = original_pose_data['side']

            """Now save the poses in the csv format that the UMI pipeline expects. This entails no longer working with poses on the trimmed video, but instead poses for the entire video. This means we need to pad the poses of the video (but setting is_lost to true so they are invalid) and later on UMI pipeline will handle selecting out the valid regions."""

            # pad timestamps to reach the full video length
            delta_t = times[1] - times[0]
            times = [times[0] - delta_t * (start_frames_to_cut - i) for i in range(start_frames_to_cut)] + list(times) # start pad timestamps
            times = times + [times[-1] + delta_t * (i + 1) for i in range(end_frames_to_cut)] # end pad timestamps
            times = np.array(times)

            # is_lost
            is_lost = [True] * start_frames_to_cut + [False] * num_frames_to_keep + [True] * end_frames_to_cut

            def save_camera_trajectory(poses, suffix=None):
                # set world frame to have the same origin as the first pose
                pos_quat_xyzw = pose_4x4_to_quat_xyzw(poses)

                # pad poses to reach the full video length
                empty_7d_pos = np.array([0,0,0,0,0,0,1]) # pos quat xyzw
                pos_quat_xyzw = np.concatenate([np.repeat(np.expand_dims(empty_7d_pos, 0), start_frames_to_cut, axis=0), pos_quat_xyzw], axis=0) # start pad poses
                pos_quat_xyzw = np.concatenate([pos_quat_xyzw, np.repeat(np.expand_dims(empty_7d_pos, 0), end_frames_to_cut, axis=0)], axis=0) # end pad poses

                poses = pos_quat_xyzw_to_4x4(pos_quat_xyzw)
                
                if suffix is None:
                    suffix = ""
                else:
                    suffix = f"_{suffix}"

                save_trajectory_umi_format(os.path.join(demonstration_dir, f'camera_trajectory{suffix}_{side}.csv'), poses, times, is_lost)

            # iPhone local frame has y up, x right and z toward you when looking at the screen with charger port on the right
            # GoPro local frame has y down, x right and z away from you when looking at the screen
            # thus converting between the two consists of a 180 degree rotation about the x axis and a translation
            I_T_G = np.array([[1,0,0,0.039711],
                              [0,-1,0,-0.008423],
                              [0,0,-1,-0.039907],
                              [0,0,0,1]]) # geometrically the pose of the GoPro in the iPhone frame; involves a 180 rotation about X axis and a translation. For iPhone 15 Pro
            
            # the poses from the iPhone (in `poses`) are respect to the world frame, so they are geometrically the transform from the world frame (which has y axis opposite of gravity) to the iPhone frame.
            # Thus the poses are W_T_I
            # We want to convert these poses to be in the GoPro frame, so we need to find the transform from the GoPro frame to the iPhone frame, which is I_T_G
            # Then we can convert the poses from the iPhone frame to the GoPro frame by W_T_G = W_T_I @ I_T_G
            gopro_cam_poses = np.array([W_T_I @ I_T_G for W_T_I in iphone_cam_poses]) # now becomes W_T_G, which is geometrically the transform from the world frame to the GoPro frame

            # save poses in two styles (both have the same ARKit world origin):
            # 1) in ARKit pose format which preserves the iPhone local coordinate frame convention from ARKit and stores the iPhone pose instead of the GoPro pose. This is useful if you want to work with iPhone poses.
            save_camera_trajectory(iphone_cam_poses, 'iphone')
            # 2) in UMI pose format which uses the GoPro/TCP local coordinate frame convention. This is useful if you want to work with GoPro poses and/or train an UMI policy.
            save_camera_trajectory(gopro_cam_poses, 'gopro')

            # save the exact timestamps of each of the GoPro frames and associated poses to the metadata
            time_strs = [datetime.fromtimestamp(x, timezone.utc).isoformat() for x in times]
            write_demonstration_metadata(demonstration_dir, {f'{side}_gopro_frame_times': time_strs})
            
        if left_present:
            left_poses = np.array(left_pose_data['poseTransforms'])
            left_num_frames, left_poses, left_times, left_start_frames_to_cut, left_end_frames_to_cut = sample_poses_at_video_frames(left_video_path, 'left', left_poses, left_pose_times, valid_start_time, valid_end_time)
        if right_present:
            right_poses = np.array(right_pose_data['poseTransforms'])
            right_num_frames, right_poses, right_times, right_start_frames_to_cut, right_end_frames_to_cut = sample_poses_at_video_frames(right_video_path, 'right', right_poses, right_pose_times, valid_start_time, valid_end_time)

        if left_present and right_present:
            assert left_num_frames == right_num_frames
            assert len(left_poses) == len(right_poses)
            time_error_ignore = left_times[0] - right_times[0]
            assert time_error_ignore <= 1/60
            print(f'Difference between left and right times that we ignore: {time_error_ignore}')

            right_times = left_times # force align them (the error is less than 1/60 of a second so it shouldn't matter too much). This force align is required because there is no way we ensure the RGB frames are perflectly aligned between the two GoPro cameras.

        if left_present:
            save_sampled(left_num_frames, left_poses, left_times, left_pose_data, left_start_frames_to_cut, left_end_frames_to_cut)
        if right_present:
            save_sampled(right_num_frames, right_poses, right_times, right_pose_data, right_start_frames_to_cut, right_end_frames_to_cut)

    print(f'\nAligned {num_processed} demonstrations')
    print(f'Already had aligned {num_already_processed} demonstrations')
