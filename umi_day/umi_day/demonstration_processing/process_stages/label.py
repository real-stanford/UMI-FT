import os
from omegaconf import DictConfig
import json
import numpy as np
from pathlib import Path
from scipy.signal import convolve
import matplotlib.pyplot as plt
from umi_day.demonstration_processing.utils.generic_util import demonstration_to_display_string, get_demonstration_sides_present, get_demonstration_video_fps, read_demonstration_metadata, get_demonstration_video_frame_count, get_demonstration_property, get_demonstration_main_video_path, get_demonstration_frame_times
from umi_day.common.timecode_util import datetime_fromisoformat
from umi_day.demonstration_processing.utils.gripper_util import get_demo_gripper_width
from dataclasses import dataclass
from typing import List
import cv2
from tqdm import tqdm

@dataclass
class Task:
    name: str
    start_idx: int
    end_idx: int

def pad_subtasks(tasks: List[Task], padding_seconds: float, capture_frame_rate:float):
    """Pad the subtasks by a fixed amount of time before the start of the subtask. This is done to make the subtasks better at transitioning from the previous subtask. Mutates `tasks` in place."""
    for task in tasks:
        new_start = int(task.start_idx - padding_seconds * capture_frame_rate)
        new_start = max(0, new_start)
        task.start_idx = new_start

def identify_tasks(demonstration_dir, cfg: DictConfig) -> List[Task]:
    """Returns a dictionary mapping from task names to (start, end) frame indices."""
    sides_present = get_demonstration_sides_present(demonstration_dir)
    assert len(sides_present) == 1, 'task extraction currently only supports demonstrations with one side'
    side = sides_present[0]

    try:
        label_type = get_demonstration_property(demonstration_dir, side, 'labelType')
    except KeyError:
        label_type = 'None'

    if label_type == 'Narration':
        tasks = identify_tasks_labeled(demonstration_dir, cfg)
    elif label_type == 'GripperWidth':
        tasks = identify_tasks_gripper_width(demonstration_dir, cfg)
    elif label_type == 'Predefined':
        tasks = identify_tasks_labeled(demonstration_dir, cfg)
    elif label_type == 'None':
        tasks = []
    else:
        raise NotImplementedError
    
    # visualize tasks
    if cfg.visualize and len(tasks) > 0:
        visualize_tasks(demonstration_dir, tasks)
    
    return tasks

def identify_tasks_labeled(demonstration_dir, cfg: DictConfig) -> List[Task]:
    """Use the narration to identify subtasks"""
    sides_present = get_demonstration_sides_present(demonstration_dir)
    assert len(sides_present) == 1, 'narration task extraction currently only supports demonstrations with one side'
    side = sides_present[0]

    episode_frame_count = get_demonstration_video_frame_count(demonstration_dir, side)

    # load the narration
    task_names = get_demonstration_property(demonstration_dir, side, 'taskNames')
    task_start_timestamps = get_demonstration_property(demonstration_dir, side, 'taskStartTimestamps')
    task_end_timestamps = get_demonstration_property(demonstration_dir, side, 'taskEndTimestamps')

    # get utc time of first RGB frame and offset the timestamps from this starting time
    first_rgb_timestamp = get_demonstration_frame_times(demonstration_dir, side)[0]
    first_rgb_timestamp = datetime_fromisoformat(first_rgb_timestamp).timestamp()
    rel_task_start_timestamps = [datetime_fromisoformat(x).timestamp() - first_rgb_timestamp for x in task_start_timestamps] # seconds relative to first rgb frame
    rel_task_end_timestamps = [datetime_fromisoformat(x).timestamp() - first_rgb_timestamp for x in task_end_timestamps] # seconds relative to first rgb frame

    # create a Task for each subtask
    tasks = []
    for i in range(len(task_names)):
        task_name = task_names[i]
        
        # convert timestamps to frame indices
        rel_task_start_timestamp = rel_task_start_timestamps[i]
        rel_task_end_timestamp = rel_task_end_timestamps[i]
        fps = get_demonstration_video_fps(demonstration_dir, side)
        start_frame_idx = max(int(rel_task_start_timestamp * fps), 0)
        end_frame_idx = min(int(rel_task_end_timestamp * fps), episode_frame_count - 1)

        if end_frame_idx != episode_frame_count - 1 and end_frame_idx > 0:
            end_frame_idx -= 1

        assert start_frame_idx >= 0 and start_frame_idx < episode_frame_count, f'Invalid start frame index {start_frame_idx}'
        assert end_frame_idx >= 0 and end_frame_idx < episode_frame_count, f'Invalid end frame index {end_frame_idx}'
        assert end_frame_idx > start_frame_idx, f'End frame index {end_frame_idx} must be greater than start frame index {start_frame_idx}'

        tasks.append(Task(task_name, start_frame_idx, end_frame_idx))
    
    return tasks

def identify_tasks_gripper_width(demonstration_dir, cfg: DictConfig) -> List[Task]:
    """Use the gripper width as a cue to identify subtasks"""
    sides_present = get_demonstration_sides_present(demonstration_dir)
    assert len(sides_present) == 1, 'gripper width task extraction currently only supports demonstrations with one side'
    side = sides_present[0]

    gripper_width = get_demo_gripper_width(demonstration_dir, side)

    def convolve_with(filter):
        # detect closing gripper
        pad_size = (len(filter) - 1) // 2
        padded_signal = np.pad(gripper_width, pad_width=pad_size, mode='edge')
        # Perform the convolution
        convolved_signal = convolve(padded_signal, filter, mode='same')
        # Truncate to the original signal length
        result = convolved_signal[pad_size:pad_size + len(gripper_width)]
        return result

    # when the gradient of the gripper width passes a threshold, we can identify gripper opening/closing
    closing_result = convolve_with(np.array([1, 1, 0, -1, -1]))
    threshold = np.abs(closing_result).max() * cfg.mode_gripper_width.gradient_threshold
    close_idx = int(np.argmax(closing_result < -threshold))
    open_idx = int(len(gripper_width) - np.argmax(closing_result[::-1] > threshold))

    # determine the subtasks
    task_id = get_demonstration_property(demonstration_dir, side, 'taskID')
    if task_id == 'cup_rearrangement':
        assert close_idx < open_idx

        tasks = [
            Task('cup_rearrangement', 0, len(gripper_width)),
            Task('orient_cup', 0, close_idx),
            Task('move_cup', close_idx, open_idx),
            Task('reset', open_idx, len(gripper_width))
        ]
    else:
        raise NotImplementedError(f'Overall task {cfg.mode_gripper_width.overall_task} not implemented')
    
    # pad the subtasks
    fps = get_demonstration_video_fps(demonstration_dir, side)
    pad_subtasks(tasks, cfg.mode_gripper_width.pad_subtask_seconds, fps)

    # visualize the subtasks
    if cfg.visualize:
        plt.ioff()
        # Create subplots
        fig, axes = plt.subplots(len(tasks), 1, sharex=True, figsize=(5, len(tasks)*1.5))
        fig.suptitle(f'\"{cfg.mode_gripper_width.overall_task}\" subtasks identified by gripper width')
        colors = ['r', 'g', 'b', 'purple']
        top = max(gripper_width) * 100 * 1.1

        for ax, (color, task) in zip(axes, zip(colors, tasks)):
            x = np.arange(task.start_idx, task.end_idx)
            y = gripper_width[task.start_idx:task.end_idx] * 100

            ax.plot(x, y, label=task, color=color)
            ax.set_ylim(bottom=0, top=top)
            ax.set_ylabel('Width (cm)')
            ax.legend(loc='lower left')
            ax.grid(True)

        # Shared x-axis label
        axes[-1].set_xlabel('Frame Index')

        # Save the figure
        out_path = Path(demonstration_dir).joinpath(f'gripper_width_task_segmentation.png')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    return tasks

def compute_progression(cfg: DictConfig, tasks: List[Task]) -> List[np.ndarray]:
    """Returns a dictionary mapping from task names to progression values."""
    progression_by_task = []
    
    for task in tasks:
        if cfg.mode == 'linear':
            progression = np.linspace(0, 1, task.end_idx - task.start_idx, dtype=np.float32)
        else:
            raise NotImplementedError(f'Progression mode {cfg.mode} not implemented')

        progression_by_task.append(progression.tolist())

    return progression_by_task


def visualize_tasks(demonstration_dir: str, tasks: List[Task]):
    sides_present = get_demonstration_sides_present(demonstration_dir)
    assert len(sides_present) == 1, 'visualization currently only supports demonstrations with one side'
    side = sides_present[0]

    video_path = get_demonstration_main_video_path(demonstration_dir, side)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    output_height = 360
    output_width = int(output_height * frame_width / cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(demonstration_dir, 'subtasks.mp4')
    fourcc = cv2.VideoWriter_fourcc('a','v','c','1')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    frame_idx = 0
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), leave=False, desc='Visualizing subtasks')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # resize frame
        resized_frame = cv2.resize(frame, (output_width, output_height))

        # find all tasks that overlap with the current frame
        matching_tasks = []
        for task in tasks:
            if frame_idx >= task.start_idx and frame_idx <= task.end_idx:
                matching_tasks.append(task)
        task_names = '\n'.join([task.name for task in matching_tasks])

        # draw task names on the frame
        font_scale = 0.8
        thickness = 2
        for i, text in enumerate(task_names.split('\n')):
            (_, label_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.putText(resized_frame, text, (5, 5+(i+1) * label_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        out.write(resized_frame)

        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()


def auto_label_demo(demonstration_dir, cfg: DictConfig):
    output_path = os.path.join(demonstration_dir, 'labels.json')

    # identify tasks
    tasks = identify_tasks(demonstration_dir, cfg.task_extraction)

    # compute task progression
    progression_by_task = compute_progression(cfg.progression, tasks)

    # store all labels in a dataframe
    out = [
        {
            'name': task.name,
            'start_idx': task.start_idx,
            'end_idx': task.end_idx,
            'labels': {'progression': progression}
        }
        for task, progression in zip(tasks, progression_by_task)] 

    with open(output_path, 'w') as f:
        json.dump(out, f)


def auto_label(demonstration_iterator, cfg: DictConfig):
    num_processed = 0
    num_skipped = 0
    already_processed = set()
    already_skipped = set()
    for demonstration_dir in demonstration_iterator('demonstration'):
        # check if we have already finished processing this demonstration
        output_path = os.path.join(demonstration_dir, 'labels.json')
        if os.path.exists(output_path) and not cfg.overwrite:
            already_skipped.add(demonstration_dir)
            num_skipped += 1
            continue
        
        print(f'Labeling {demonstration_to_display_string(demonstration_dir)}')
        
        auto_label_demo(demonstration_dir, cfg)

        already_processed.add(demonstration_dir)
        num_processed += 1

    print(f'\nLabeled {num_processed} demonstrations')
    print(f'Skipped {num_skipped} demonstrations')
