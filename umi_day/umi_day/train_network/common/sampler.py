"""Modified from UMI codebase to support returning start, current, end indices of the episode."""

from typing import Optional
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from umi_day.common.replay_buffer import ReplayBuffer

def get_val_mask(n_segments, val_ratio, seed=0):
    """If sampling over tasks, then `n_segments` is the number of tasks. If sampling over episodes, then `n_segments` is the number of episodes."""
    val_mask = np.zeros(n_segments, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_segments * val_ratio)), n_segments-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_segments, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        max_duration: Optional[float]=None,
        sample_type='episode', # 'episode' or 'task'
        labels_keys: Optional[list]=None,
        max_segments: int=-1
    ):
        assert sample_type in ['episode', 'task']
        if sample_type == 'episode':
            assert not labels_keys, 'cannot specify `labels_keys` when sampling over episodes'
        segment_ends = replay_buffer.episode_ends[:] if sample_type == 'episode' else replay_buffer.task_data_ends[:]

        if sample_type == 'task':
            task_to_episode_idxs = replay_buffer.get_task_to_episode_idxs()

        # create indices, including (current_within_segment_idx, episode_idx, task_idx)
        start_of_segment_indices = {}
        end_of_segment_indices = {}
        indices = list()
        for i in range(len(segment_ends)):
            # TODO: could clean this up since there is a lot of indexing work that should be done only once (and not again in `sample_sequence` and can be offload to the replay_buffer for consistency)
            if len(start_of_segment_indices) == max_segments:
                break
            
            if mask is not None and not mask[i]:
                # skip episode/task
                continue
            end_data_idx = segment_ends[i]
            if sample_type == 'task':
                segment_length = replay_buffer.task_lengths[i]
                start_data_idx = end_data_idx - segment_length
            else:
                start_data_idx = 0 if i == 0 else segment_ends[i-1]
                segment_length = end_data_idx - start_data_idx
            if max_duration is not None:
                end_data_idx = start_data_idx + min(end_data_idx-start_data_idx, max_duration * 60)

            start_of_segment_indices[i] = len(indices)
            for current_data_idx in range(start_data_idx, end_data_idx):
                if not action_padding and end_data_idx < current_data_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                
                current_within_segment_idx = current_data_idx - start_data_idx

                if sample_type == 'task':
                    task_idx = i
                    episode_idx = task_to_episode_idxs[task_idx]
                else:
                    task_idx = None
                    episode_idx = i
                
                indices.append((current_within_segment_idx, episode_idx, task_idx))
            end_of_segment_indices[i] = len(indices)
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1

            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)

        # copy labels to memory
        for key in labels_keys:
            self.replay_buffer[key] = replay_buffer.labels[key][:]

        self.action_padding = action_padding
        self.indices = indices
        self.start_of_segment_indices = start_of_segment_indices
        self.end_of_segment_indices = end_of_segment_indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.labels_keys = labels_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.sample_type = sample_type

        self.orig_replay_buffer = replay_buffer
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        current_within_segment_idx, episode_idx, task_idx = self.indices[idx]
        result = dict()

        if self.sample_type == 'task':
            task_length = self.orig_replay_buffer.task_lengths[task_idx]
            result['task_idx'] = task_idx
            
            # data idx
            end_data_idx = self.orig_replay_buffer.task_data_ends[task_idx]
            start_data_idx = end_data_idx - task_length
            current_data_idx = start_data_idx + current_within_segment_idx

            # task idx
            start_labels_idx = self.orig_replay_buffer.task_labels_ends[task_idx] - task_length
            end_labels_idx = self.orig_replay_buffer.task_labels_ends[task_idx]
            current_labels_idx = start_labels_idx + current_within_segment_idx
        else: # episode
            data_slice = self.orig_replay_buffer.get_episode_slice(episode_idx)
            start_data_idx, end_data_idx = data_slice.start, data_slice.stop
            current_data_idx = start_data_idx + current_within_segment_idx

        result['episode_idx'] = episode_idx

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            is_key_upsampled = self.orig_replay_buffer.is_key_upsampled(key)
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0

                if is_key_upsampled:
                    # if key is upsampled, then we need to handle the horizon differently. In particular, we apply the horizon and downsampling at the frequency of the observed data.
                    # so if we have 10Hz ultrawide data (and 60Hz other data), then a horizon of 2 will find the two most recent ultrawide frames (note these frames will be different)
                    # this is different than just duplicating the 10Hz data to make it 60Hz and then getting the last 2 frames becuase in that case the two frames would likely be the same, while in our implementation we ensure those are two different frames since we operate on the sampling frequency of 10Hz (rather than 60Hz upsampled stream of the data)
                    upsample_current_data_idx = self.orig_replay_buffer.map_upsample_index(key, current_data_idx)
                    upsample_start_data_idx = self.orig_replay_buffer.map_upsample_index(key, start_data_idx)
                    num_valid = min(this_horizon, (upsample_current_data_idx - upsample_start_data_idx) // this_downsample_steps + 1)
                    upsample_slice_start = upsample_current_data_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[upsample_slice_start: upsample_current_data_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
                else:
                    num_valid = min(this_horizon, (current_data_idx - start_data_idx) // this_downsample_steps + 1)
                    slice_start = current_data_idx - (num_valid - 1) * this_downsample_steps

                    output = input_arr[slice_start: current_data_idx + 1: this_downsample_steps]
                    assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                assert not is_key_upsampled, 'upsampling not yet supported for low dim data'
                idx_with_latency = np.array(
                    [current_data_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_data_idx, end_data_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_data_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_data_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)
                
            result[key] = output

        # action
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_data_idx, current_data_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_data_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        # labels
        if self.sample_type == 'task' and self.labels_keys:
            result['labels'] = {}
            for key in self.labels_keys:
                result['labels'][key] = self.replay_buffer[key][current_labels_idx]

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
