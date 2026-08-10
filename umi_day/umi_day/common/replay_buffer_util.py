"""Utility to print out the contents of a replay buffer for debugging."""

import argparse
import zarr
from umi_day.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
import cv2
import os
import numpy as np
from tqdm import tqdm

def print_replay_buffer(path, vis_iphone_rgb: bool = False, vis_iphone_video: bool = False):
    register_codecs(verbose=False)
    with zarr.ZipStore(path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )

    print(replay_buffer)
    print(f'Task names: {replay_buffer.task_names[:]}')

    out_dir = os.path.dirname(path)
    if vis_iphone_rgb:
        # single frame of main camera
        sample_index = np.random.randint(0, replay_buffer['camera0_main_rgb'].shape[0])
        main_rgb = replay_buffer['camera0_main_rgb'][sample_index][...,::-1]
        out_path = os.path.join(out_dir, 'tmp_main_rgb.png')
        cv2.imwrite(out_path, main_rgb)
        print(f'Saved main RGB image to {out_path}')

        # single frame of ultrawide camera
        ultrawide_sample_index = replay_buffer.map_upsample_index('camera0_ultrawide_rgb', sample_index)
        ultrawide_rgb = replay_buffer['camera0_ultrawide_rgb'][ultrawide_sample_index][...,::-1]
        out_path = os.path.join(out_dir, 'tmp_ultrawide_rgb.png')
        cv2.imwrite(out_path, ultrawide_rgb)
        print(f'Saved ultrawide RGB image to {out_path}')

    if vis_iphone_video:
        # video of main camera
        video_path = os.path.join(out_dir, 'tmp_main_video.mp4')
        frame_width = replay_buffer['camera0_main_rgb'].shape[2]
        frame_height = replay_buffer['camera0_main_rgb'].shape[1]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 60, (frame_width, frame_height))

        for i in tqdm(range(replay_buffer['camera0_main_rgb'].shape[0]), desc='Writing main video'):
            frame = replay_buffer['camera0_main_rgb'][i][...,::-1]
            out.write(frame)

        out.release()
        print(f'Saved main video to {video_path}')

        # video of ultrawide camera
        video_path = os.path.join(out_dir, 'tmp_ultrawide_video.mp4')
        frame_width = replay_buffer['camera0_ultrawide_rgb'].shape[2]
        frame_height = replay_buffer['camera0_ultrawide_rgb'].shape[1]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('a','v','c','1'), 10, (frame_width, frame_height))

        for i in tqdm(range(replay_buffer['camera0_ultrawide_rgb'].shape[0]), desc='Writing ultrawide video'):
            frame = replay_buffer['camera0_ultrawide_rgb'][i][...,::-1]
            out.write(frame)

        out.release()
        print(f'Saved ultrawide video to {video_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View the contents of a replay buffer.')
    parser.add_argument('dataset_path', type=str, help='Path to the replay buffer to view.')
    parser.add_argument('--vis_iphone_rgb', action='store_true', help='Save images of the iPhone RGB data.')
    parser.add_argument('--vis_iphone_video', action='store_true', help='Save video of the iPhone RGB data.')
    args = parser.parse_args()

    print_replay_buffer(args.dataset_path, vis_iphone_rgb=args.vis_iphone_rgb, vis_iphone_video=args.vis_iphone_video)
