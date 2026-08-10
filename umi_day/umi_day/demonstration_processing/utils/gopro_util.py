from datetime import timezone, timedelta
from umi_day.common.timecode_util import mp4_get_start_datetime

def get_gopro_start_video_time(path, gopro_latency_correction_ms=0):
    # and logic to add correction offset to align GoPro with QR code
    date = mp4_get_start_datetime(path) + timedelta(milliseconds=gopro_latency_correction_ms)

    # at this point we know this data is in UTC time so attach UTC timezone to the timestamp
    date = date.replace(tzinfo=timezone.utc)

    return date
