from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import glob
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from umi_day.common.replay_buffer import ReplayBuffer
from umi_day.train_network.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_identity_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
)
from typing import Optional
import torch.nn.functional as F

register_codecs()
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from ..model.language.language_embedding import LanguageEmbedding

from libero.libero.envs.bddl_utils import get_problem_info
from umi_day.train_network.utils.libero_util import hdf5_to_bddl
from umi_day.train_network.common.augmentation import ImageAugmentation

class LiberoReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        language_embedding: LanguageEmbedding,
        action_padding: bool=False,
        max_duration: Optional[float]=None,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=False,
        use_cache=False,
        overwrite_cache=False,
        seed=42,
        val_ratio=0.0,
        sample_type:str='episode',
        max_segments:int=-1,
        name_suffix:str='',
        include_file_filters: list[str]=[],
        train_image_augmentation: ImageAugmentation=None
    ):

        rotation_transformer = RotationTransformer(
            from_rep="axis_angle", to_rep=rotation_rep
        )

        replay_buffer = None
        if use_cache:
            file_name = os.path.join('cache', 'libero', os.path.basename(dataset_path))
            if name_suffix:
                file_name = f"{file_name}_{name_suffix}"
            cache_zarr_path = file_name + ".zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            print("Cache path:", cache_zarr_path)

            with FileLock(cache_lock_path):
                if overwrite_cache and os.path.exists(cache_zarr_path):
                    print("Overwriting cache.")
                    os.remove(cache_zarr_path)
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")

                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            rotation_transformer=rotation_transformer,
                            include_file_filters=include_file_filters
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            os.remove(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                rotation_transformer=rotation_transformer,
                include_file_filters=include_file_filters
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        val_mask = get_val_mask(
            n_segments=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask

        language_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            is_language = attr.get('is_language', False)
            if is_language:
                language_keys.append(key)
                continue

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

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key and not shape_meta['obs'][key].get('is_language', False):
                self.sampler_lowdim_keys.append(key)

        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.language_keys = language_keys
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.action_padding = action_padding
        self.max_duration = max_duration
        self.train_mask = train_mask
        self.use_legacy_normalizer = use_legacy_normalizer
        self.key_horizon = key_horizon
        self.key_down_sample_steps = key_down_sample_steps
        self.key_latency_steps = key_latency_steps
        self.sample_type = sample_type
        self.labels_keys = []
        self.max_segments = max_segments
        self.include_file_filters = include_file_filters
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
            mask=~self.train_mask,
            action_padding=self.action_padding,
            max_duration=self.max_duration,
            sample_type=self.sample_type,
            labels_keys=self.labels_keys,
            max_segments=self.max_segments
        )

        val_set.train_mask = ~self.train_mask
        val_set.is_training_dataset = False
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action normalizer
        stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = get_identity_normalizer_from_stat(stat)

        # lowdim normalizer (excluding language)
        for key in self.lowdim_keys:
            if key in self.language_keys:
                continue

            stat = array_to_stats(self.replay_buffer[key])

            if key == "ee_pos": # ee_pos is absolute eposition
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "ee_ori": # ee_ori is axis angle format
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key == "gripper_states": # gripper_stages is [0 to 0.04, -0.04 to 0]
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("language"):
                continue  ## skip
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # language normalizer
        for key in self.language_keys:
            language_entry = self[0]['obs'][key]
            stat = {
                'min': np.zeros_like(language_entry),
                'max': np.zeros_like(language_entry),
            }
            normalizer[key] = get_identity_normalizer_from_stat(stat)

        # image normalizer
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            obs_dict[key] = np.rot90(obs_dict[key], k=2, axes=(2, 3)).copy()
            obs_dict[key] = np.flip(obs_dict[key], axis=3).copy()

            # resize image to be 224x224
            resize = 224
            obs_dict[key] = F.interpolate(
                torch.tensor(obs_dict[key], dtype=torch.float32),
                size=(resize, resize),
                mode="bilinear",
                align_corners=False,
            ).numpy()

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

        # additional metadata for debugging
        metadata = {
            'episode_idx': torch.tensor(data['episode_idx'], dtype=torch.int64)
        }

        # add task_idx if using task sampling
        if self.sample_type == 'task':
            metadata['task_idx'] = torch.tensor(data['task_idx'], dtype=torch.int64)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
            'metadata': metadata
        }

        # apply image augmentations if training dataset
        if self.is_training_dataset and self.train_image_augmentation is not None:
            torch_data['obs'] = self.train_image_augmentation.apply(torch_data['obs'])

        return torch_data


def _convert_actions(raw_actions, rotation_transformer):
    """Convert the rotation in the actions according to `rotation_transformer`."""
    pos = raw_actions[..., :3]
    rot = raw_actions[..., 3:6]
    gripper = raw_actions[..., 6:]
    rot = rotation_transformer.forward(rot)
    raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)
    return raw_actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    include_file_filters=[]
):
    """The obs keys in the demos are: ['agentview_rgb', 'ee_pos', 'ee_ori' (axis angle format), 'ee_states' (ee_pos concat with ee_ori), 'eye_in_hand_rgb', 'gripper_states' , 'joint_states']"""
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        shape = attr["shape"]
        type = attr.get("type", "low_dim")
        if type == "rgb":
            rgb_keys.append(key)
        elif type == "low_dim":
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    file_handles = []  # Store file handles if you need to keep them open
    demos_all = {}
    language_all = {}
    count = 0
    count_i_to_relative_i = {}
    
    dataset_paths = glob.glob(dataset_path + "/*.hdf5")

    if include_file_filters:
        dataset_paths = [x for x in dataset_paths if os.path.basename(x).replace('.hdf5', '') in include_file_filters]

    for dataset_path_each in dataset_paths:
        cur_bddl_path = hdf5_to_bddl(dataset_path_each)
        problem_info = get_problem_info(cur_bddl_path)
        language_goal = problem_info["language_instruction"]

        print(f"Loading {dataset_path_each}")
        file = h5py.File(
            dataset_path_each, "r"
        )  # Open the file without closing it immediately
        file_handles.append(
            file
        )  # Keep track of the file handle to avoid it being closed
        demos = file["data"]

        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            demos_all[f"demo_{count}"] = demo
            language_all[f"demo_{count}"] = language_goal
            count_i_to_relative_i[count] = i
            count += 1
    print("Total demos:", count)

    demos = demos_all
    episode_ends = list()
    task_lengths = list()
    episode_names = list()
    prev_end = 0
    for i in range(len(demos)):
        demo_i_str = f"demo_{i}"
        relative_demo_i_str = f"demo_{count_i_to_relative_i[i]}"
        demo = demos[demo_i_str]
        episode_length = demo["actions"].shape[0]
        episode_end = prev_end + episode_length
        prev_end = episode_end
        episode_ends.append(episode_end)
        task_lengths.append(episode_length)
        episode_names.append(f"{language_all[demo_i_str]} - {relative_demo_i_str}")
    n_steps = episode_ends[-1]
    episode_starts = [0] + episode_ends[:-1]
    _ = meta_group.array(
        "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )
    _ = meta_group.array(
        "episode_names", episode_names, dtype=str, compressor=None, overwrite=True
    )
    _ = meta_group.array(
        "task_lengths", task_lengths, dtype=np.int64, compressor=None, overwrite=True
    )
    _ = meta_group.array(
        "task_data_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )
    _ = meta_group.array(
        "task_data_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )
    _ = meta_group.array(
        "task_labels_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
    )

    # save lowdim data
    for key in tqdm(lowdim_keys + ["action"], desc="Loading lowdim data"):
        data_key = "obs/" + key
        if key == "action":
            data_key = "actions"
            this_language_data = list()
        if key == "task_language":
            continue
        this_data = list()
        for i in range(len(demos)):
            demo = demos[f"demo_{i}"]
            this_data.append(demo[data_key][:].astype(np.float32))

            if key == "action":
                this_language_data.append(language_all[f"demo_{i}"])

        this_data = np.concatenate(this_data, axis=0)

        if key == "action":
            this_data = _convert_actions(
                raw_actions=this_data,
                rotation_transformer=rotation_transformer,
            )

            assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])

            this_language_data = np.array(this_language_data)
        else:
            assert this_data.shape == (n_steps,) + tuple(
                shape_meta["obs"][key]["shape"]
            )
        _ = data_group.array(
            name=key,
            data=this_data,
            shape=this_data.shape,
            chunks=this_data.shape,
            compressor=None,
            dtype=this_data.dtype,
        )

        if key == "action":
            _ = meta_group.array(
                name="task_names",
                data=this_language_data,
                shape=this_language_data.shape,
                chunks=this_language_data.shape,
                compressor=None,
                dtype=this_language_data.dtype,
            )

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    with tqdm(
        total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
    ) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                data_key = "obs/" + key
                shape = (3, 128, 128)
                c, h, w = shape
                this_compressor = Jpeg2k(level=50)
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=this_compressor,
                    dtype=np.uint8,
                )

                for episode_idx in range(len(demos)):
                    demo = demos[f"demo_{episode_idx}"]
                    hdf5_arr = demo["obs"][key]
                    for hdf5_idx in range(hdf5_arr.shape[0]):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(completed))

                        zarr_idx = episode_starts[episode_idx] + hdf5_idx
                        futures.add(
                            executor.submit(
                                img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                            )
                        )
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(completed))

    # Ensure you close all files when you're done with them
    for file in file_handles:
        file.close()

    # Add missing fields in
    _ = root.require_group("labels", overwrite=True)

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
