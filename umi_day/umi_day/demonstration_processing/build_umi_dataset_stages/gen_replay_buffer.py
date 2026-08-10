# adapted from 07_generate_replay_buffer.py from UMI

import sys
import os

# %%
from pathlib import Path
import zarr
import pickle
import numpy as np
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from omegaconf import DictConfig
from umi_day.common.cv_util import get_image_transform_with_border
from umi_day.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def gen_replay_buffer(session_dir: str, dataset_plan_path: str, out_replay_buffer_path: str, cfg: DictConfig):
    if cfg.num_workers == -1:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = cfg.num_workers

    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    n_grippers = -1
    n_cameras = -1
    buffer_start_main = 0
    buffer_start_ultrawide = 0
    vid_args = []
    demos_path = Path(session_dir).joinpath('demos')
    
    with open(dataset_plan_path, 'rb') as f:
        plan = pickle.load(f)
    
    """ dump lowdim data to replay buffer """
    for plan_episode in plan:
        grippers = plan_episode['grippers']
        
        # check that all episodes have the same number of grippers
        if n_grippers == -1:
            n_grippers = len(grippers)
        else:
            assert n_grippers == len(grippers)
            
        # check that all episodes have the same number of cameras
        cameras = plan_episode['cameras']
        if n_cameras == -1:
            n_cameras = len(cameras)
        else:
            assert n_cameras == len(cameras)
        
        # low dim episode data
        episode_data = dict()
        upsample_indexing_values = dict()
        upsample_indexing_lengths = dict()
        for gripper_id, gripper in enumerate(grippers):    
            eef_pose = gripper['tcp_pose']
            eef_pos = eef_pose[...,:3]
            eef_rot = eef_pose[...,3:]
            gripper_widths = gripper['gripper_width']
            demo_start_pose = np.empty_like(eef_pose)
            demo_start_pose[:] = gripper['demo_start_pose']
            demo_end_pose = np.empty_like(eef_pose)
            demo_end_pose[:] = gripper['demo_end_pose']
            
            robot_name = f'robot{gripper_id}'
            episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
            episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
            episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
            episode_data[robot_name + '_demo_start_pose'] = demo_start_pose.astype(np.float32)
            episode_data[robot_name + '_demo_end_pose'] = demo_end_pose.astype(np.float32)

        # upsample indexing for ultrawide
        for camera_idx, camera in enumerate(cameras):
            data_key_name = f'camera{camera_idx}_ultrawide_rgb'
            indexing_key = 'main_idx_to_ultrawide_idx'
            upsample_indexing_values[data_key_name] = camera[indexing_key]
            start, end = camera['ultrawide_video_start_end']
            upsample_indexing_lengths[data_key_name] = end - start

        episode_name = plan_episode['episode_name']
        
        vid_metadata = plan_episode['tasks']
        
        out_replay_buffer.add_episode(data=episode_data, tasks=vid_metadata, compressors=None, episode_name=episode_name, upsample_indexing_values=upsample_indexing_values, upsample_indexing_lengths=upsample_indexing_lengths)
        
        # aggregate video gen arguments
        n_frames_main = -1
        n_frames_ultrawide = -1
        for cam_id, camera in enumerate(cameras):
            main_video_path = demos_path.joinpath(camera['main_video_path']).absolute()
            ultrawide_video_path = demos_path.joinpath(camera['ultrawide_video_path']).absolute()
            assert main_video_path.is_file()
            assert ultrawide_video_path.is_file()
            
            # for main camera we can assert that each camera has the same number of frames
            main_video_start, main_video_end = camera['main_video_start_end']
            if n_frames_main == -1:
                n_frames_main = main_video_end - main_video_start
            else:
                assert n_frames_main == (main_video_end - main_video_start)

            vid_args.append({
                'video_path': str(main_video_path),
                'camera_idx': cam_id,
                'frame_start': main_video_start,
                'frame_end': main_video_end,
                'buffer_start': buffer_start_main,
                'type': 'main_rgb',
            })

            # for the ultrawide camera, we can't perform this same assumption becuase it's sampled at a lower fequency (10Hz), so it's possible that each camera has a different number of frames (by at most a difference of 1). TODO: to support bimanual we will need to handle this case for the buffer sizes
            ultrawide_video_start, ultrawide_video_end = camera['ultrawide_video_start_end']
            if n_frames_ultrawide == -1:
                n_frames_ultrawide = ultrawide_video_end - ultrawide_video_start
            else:
                assert abs(n_frames_ultrawide - (ultrawide_video_end - ultrawide_video_start)) <= 1
            
            vid_args.append({
                'video_path': str(ultrawide_video_path),
                'camera_idx': cam_id,
                'frame_start': ultrawide_video_start,
                'frame_end': ultrawide_video_end,
                'buffer_start': buffer_start_ultrawide,
                'type': 'ultrawide_rgb',
            })

        buffer_start_main += n_frames_main
        buffer_start_ultrawide += n_frames_ultrawide
    
    """ Add videos to replay buffer """
    out_res = (224, 224)
    num_episodes = len(plan)
    
    print(f"{num_episodes} episodes used in total ({len(vid_args)} videos)!")
    
    # get image size
    assert vid_args[0]['type'] == 'main_rgb'
    with av.open(vid_args[0]['video_path']) as container:
        in_stream = container.streams.video[0]
        main_ih, main_iw = in_stream.height, in_stream.width
    
    assert vid_args[1]['type'] == 'ultrawide_rgb'
    with av.open(vid_args[1]['video_path']) as container:
        in_stream = container.streams.video[0]
        ultrawide_ih, ultrawide_iw = in_stream.height, in_stream.width

    # image arrays
    img_compressor = JpegXl(level=99, numthreads=1)
    for cam_id in range(n_cameras):
        main_name = f'camera{cam_id}_main_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=main_name,
            shape=(buffer_start_main,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )
        ultrawide_name = f'camera{cam_id}_ultrawide_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=ultrawide_name,
            shape=(buffer_start_ultrawide,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, vid_metadata):
        vid_type = vid_metadata['type']

        if vid_type == 'main_rgb':
            iw, ih = main_iw, main_ih
        elif vid_type == 'ultrawide_rgb':
            iw, ih = ultrawide_iw, ultrawide_ih

        resize_tf = get_image_transform_with_border(
            in_res=(iw, ih),
            out_res=out_res
        )
        camera_idx = vid_metadata['camera_idx']
        name = f'camera{camera_idx}_{vid_type}'
        img_array = replay_buffer.data[name]
        
        with av.open(vid_metadata['video_path']) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if frame_idx < vid_metadata['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < vid_metadata['frame_end']:
                    if frame_idx == vid_metadata['frame_start']:
                        buffer_idx = vid_metadata['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # resize
                    img = resize_tf(img)
                    
                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                else:
                    break
                    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for vid_metadata in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, vid_metadata))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    # dump to disk
    print(f"Saving ReplayBuffer to {out_replay_buffer_path}")
    with zarr.ZipStore(out_replay_buffer_path, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {num_episodes} episodes used in total!")
