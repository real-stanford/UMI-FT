from omegaconf import DictConfig
import json
import os
from datetime import timedelta
import cv2
import tqdm
from statistics import stdev
from umi_day.demonstration_processing.utils.gopro_util import get_gopro_start_video_time
from umi_day.demonstration_processing.utils.generic_util import write_demonstration_metadata
from umi_day.common.timecode_util import datetime_fromisoformat
from umi_day.demonstration_processing.utils.generic_util import demonstration_to_display_string, get_demonstration_video_frame_count, get_demonstration_video_fps

def compute_gopro_ahead_of_qr_ms(video_path, max_detections=10):
    video_start_time = get_gopro_start_video_time(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = cv2.QRCodeDetector()

    progress_bar = tqdm.tqdm(total=max_detections)
    i = 0
    num_detections = 0
    last_detection = ""
    ms_gopro_ahead_of_qr = 0
    video_cur_time = video_start_time
    all_ms_deltas = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        decoded_text, _, _ = detector.detectAndDecodeCurved(frame)
        if decoded_text is not None and decoded_text != '' and last_detection != decoded_text:
            # ignore frames which have the same QR code as the last frame
            num_detections += 1
            qr_time = datetime_fromisoformat(decoded_text)
            msDelta = (video_cur_time - qr_time).total_seconds() * 1000
            all_ms_deltas.append(msDelta)
            ms_gopro_ahead_of_qr = (ms_gopro_ahead_of_qr * (num_detections-1) + msDelta) / num_detections

            last_detection = decoded_text
            progress_bar.update(1)

        i += 1
        stdev_txt = f'{stdev(all_ms_deltas):.2f}' if len(all_ms_deltas) > 1 else 'N/A'
        progress_bar.set_description(f'{num_detections}/{i} had QR detected. Last detect on frame {i}. GoPro ahead of QR by {int(ms_gopro_ahead_of_qr)}ms with stdev {stdev_txt}ms')
        video_cur_time = video_start_time + timedelta(seconds=i/fps)

        if num_detections >= max_detections:
            break

    cap.release()
    cv2.destroyAllWindows()

    return ms_gopro_ahead_of_qr

def gopro_timesync(demonstration_iterator, cfg: DictConfig):
    num_processed = 0
    num_already_processed = 0
    for demonstration_dir in demonstration_iterator('qrcalibration'):
        left_video_path = os.path.join(demonstration_dir, 'left.mp4')
        right_video_path = os.path.join(demonstration_dir, 'right.mp4')

        def process_side(video_path, side):
            nonlocal num_processed, num_already_processed
            if os.path.exists(video_path):
                key_name = f'msGoProAheadOfQR'

                out_path = os.path.join(demonstration_dir, f'processed_{side}.json')
                if os.path.exists(out_path) and not cfg.overwrite:
                    with open(out_path, 'r') as f:
                        data = json.load(f)
                    with open(os.path.join(demonstration_dir, f'{side}.json'), 'r') as f:
                        iphone_data = json.load(f)
                    duration = get_demonstration_video_frame_count(demonstration_dir, side) / get_demonstration_video_fps(demonstration_dir, side)
                    print(f'Skipping {demonstration_to_display_string(demonstration_dir, side)} because output already exists. GoPro latency: {data[key_name]}ms. iPhone latency: {iphone_data["timeIPhoneAheadOfQRinMS"]}ms. Video duration: {duration:.2f}s.')
                    num_already_processed += 1
                    return

                print(f'[{demonstration_dir}] Computing GoPro QR alignment for {side} side')
                gopro_ahead_of_qr_ms = int(compute_gopro_ahead_of_qr_ms(video_path))
                with open(out_path, 'w') as f:
                    json.dump({'msGoProAheadOfQR': gopro_ahead_of_qr_ms}, f)

                write_demonstration_metadata(demonstration_dir, {f'{side}_{key_name}': gopro_ahead_of_qr_ms})
                num_processed += 1

        process_side(left_video_path, 'left')
        process_side(right_video_path, 'right')
    
    print(f'\nPerformed timesync on {num_processed} demonstrations')
    print(f'Already had performed timesync on {num_already_processed} demonstrations')
