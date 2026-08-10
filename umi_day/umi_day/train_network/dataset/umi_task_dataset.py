import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil
from typing import Callable, List

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from umi_day.common.replay_buffer import ReplayBuffer
from umi_day.train_network.common.sampler import get_val_mask
from umi_day.train_network.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d
from ..model.language.language_embedding import LanguageEmbedding
from umi_day.train_network.common.augmentation import ImageAugmentation

register_codecs()

class UmiTaskDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        language_embedding: LanguageEmbedding,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        include_labels: bool=True,
        sample_type: str='task',
        max_segments: int=-1,
        train_image_augmentation: ImageAugmentation=None
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        language_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            is_language = attr.get('is_language', False)
            if is_language:
                language_keys.append(key)
                lowdim_keys.append(key)
                continue
            elif type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        n_segments = replay_buffer.n_episodes if sample_type == 'episode' else replay_buffer.n_tasks
        if max_segments > 0:
            n_segments = min(n_segments, max_segments)
        val_mask = get_val_mask(
            n_segments=n_segments,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key and not shape_meta['obs'][key].get('is_language', False):
                self.sampler_lowdim_keys.append(key)
    
        # if sampling episodes, then add episode start and end pose to observation
        if sample_type == 'episode':
            for key in replay_buffer.keys():
                if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                    self.sampler_lowdim_keys.append(key)
                    query_key = key.split('_')[0] + '_eef_pos'
                    key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                    key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                    key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        # determine label keys
        labels_keys = []
        if include_labels:
            assert sample_type == 'task'
            labels_shape_meta = shape_meta['labels']
            labels_keys = list(labels_shape_meta.keys())

        # determine unique task names
        task_names = replay_buffer.task_names
        unique_task_names = list(set(task_names))
        task_idx_to_unique_task_idx = []
        for task_name in task_names:
            task_idx_to_unique_task_idx.append(unique_task_names.index(task_name))
        task_idx_to_unique_task_idx = np.array(task_idx_to_unique_task_idx)

        # compute language embedding
        self.language_key_to_embedding = {}
        for language_key in language_keys:
            if language_key == 'task_language':
                assert sample_type == 'task'
                self.language_key_to_embedding[language_key] = language_embedding.embed(replay_buffer.task_names)
            else:
                raise NotImplementedError(f'language key `{language_key}` not supported')

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.language_keys = language_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.max_duration = max_duration
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.include_labels = include_labels
        self.sample_type = sample_type
        self.labels_keys = labels_keys
        self.max_segments = max_segments
        self.unique_task_names = unique_task_names
        self.task_idx_to_unique_task_idx = task_idx_to_unique_task_idx
        self.train_image_augmentation = train_image_augmentation
        self.is_training_dataset = True

        sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            mask=train_mask,
            action_padding=self.action_padding,
            max_duration=self.max_duration,
            sample_type=self.sample_type,
            labels_keys=self.labels_keys,
            max_segments=self.max_segments
        )
        self.sampler = sampler

    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            mask=self.val_mask,
            action_padding=self.action_padding,
            max_duration=self.max_duration,
            sample_type=self.sample_type,
            labels_keys=self.labels_keys,
            max_segments=self.max_segments
        )

        val_set.is_training_dataset = False
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            if key in self.language_keys:
                continue
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # language observations
        for key in self.language_keys:
            stat = array_to_stats(data_cache[key])
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        # language encoding
        for key in self.language_keys:
            language_embedding = self.language_key_to_embedding[key]
            if key == 'task_language':
                batch_language_embedding = language_embedding[data['task_idx']]
            else:
                raise NotImplementedError(f'language key `{key}` not supported')
            batch_language_embedding = np.expand_dims(batch_language_embedding, axis=0) # (D) -> (horizon=1, D)
            obs_dict[key] = batch_language_embedding
        
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
                
        # generate relative pose with respect to episode or task start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f'robot{other_robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                (f'robot{other_robot_id}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                continue
            
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose for the start of the episode/task
            if self.sample_type == 'episode':
                start_pose = obs_dict[f'robot{robot_id}_demo_start_pose'][0] # entire array should be the same since we sample within the episode range, so we just take the first element
            elif self.sample_type == 'task':
                # not explicitly stored in replay buffer, but we can just compute it
                task_idx = data['task_idx']
                task_data_start_idx = self.replay_buffer.task_data_ends[task_idx] - self.replay_buffer.task_lengths[task_idx]
                start_pos = self.replay_buffer[f'robot{robot_id}_eef_pos'][task_data_start_idx]
                start_rot_axis_angle = self.replay_buffer[f'robot{robot_id}_eef_rot_axis_angle'][task_data_start_idx]
                start_pose = np.concatenate([start_pos, start_rot_axis_angle])
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        # compute action and eef pose in requested (likely relative) representation (starting from absolute)
        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])
            
            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
        
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
            
        data['action'] = np.concatenate(actions, axis=-1)

        # additional metadata for debugging
        metadata = {
            'episode_idx': torch.tensor(data['episode_idx'], dtype=torch.int64)
        }

        # add task_idx if using task sampling
        if self.sample_type == 'task':
            metadata['task_idx'] = torch.tensor(data['task_idx'], dtype=torch.int64)
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32)),
            'metadata': metadata
        }

        # apply image augmentations if training dataset
        if self.is_training_dataset and self.train_image_augmentation is not None:
            torch_data['obs'] = self.train_image_augmentation.apply(torch_data['obs'])

        if self.include_labels:
            torch_data['labels'] = dict_apply(data['labels'], torch.from_numpy)

        return torch_data
