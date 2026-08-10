from typing import List, Optional, Union, Dict, Callable
import copy
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi_day.deployment.arx.peripherals.iphone_camera import IPhoneCamera
from peripherals.video_recorder import VideoRecorder
from collections import OrderedDict
from umi_day.deployment.iphone.iphone_deploy import run_server_on_thread


class MultiIPhoneCamera:
    def __init__(
        self,
        # v4l2 device file path
        server_ip: str,
        iphone_ports: List[int],
        stream_names: List[str] = ["main_rgb", "ultrawide_rgb"],
        shm_manager: Optional[SharedMemoryManager] = None,
        put_fps=None,
        put_downsample=True,
        get_max_k=30,
        receive_latency=0.0,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]] = None,
        recording_transform: Optional[
            Union[Callable[[Dict], Dict], List[Callable]]
        ] = None,
        verbose=False,
    ):
        super().__init__()

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        n_cameras = len(iphone_ports) * 2 # multiply by 2 to account for main and ultrawide

        transform = repeat_to_list(transform, n_cameras, Callable)
        vis_transform = repeat_to_list(vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(recording_transform, n_cameras, Callable)

        cameras = OrderedDict()
        for i, port in enumerate(iphone_ports):
            for stream_name in stream_names:
                if stream_name == "main_rgb":
                    port_offset = 0
                elif stream_name == "ultrawide_rgb":
                    port_offset = 1
                else:
                    raise ValueError(f"Unknown stream name: {stream_name}")

                video_recorder = VideoRecorder.create_h264(
                    fps=60, input_pix_fmt="rgb24"
                )
                camera_name = f"camera{i}_{stream_name}"
                cameras[camera_name] = IPhoneCamera(
                    name=camera_name,
                    shm_manager=shm_manager,
                    server_ip=server_ip,
                    server_port=port + port_offset,
                    resolution=(320, 240),
                    capture_fps=60, # TODO: perhaps adjust for 10Hz ultrawide
                    put_fps=put_fps,
                    put_downsample=put_downsample,
                    get_max_k=get_max_k,
                    receive_latency=receive_latency,
                    transform=transform[i],
                    vis_transform=vis_transform[i],
                    recording_transform=recording_transform[i],
                    video_recorder=video_recorder,
                    verbose=verbose,
                )

        self.cameras: Dict[str, IPhoneCamera] = cameras
        self.shm_manager = shm_manager

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)

        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def wait_until_buffer_full(self):
        for camera in self.cameras.values():
            camera.wait_until_buffer_full()

    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Returns:
        {
            0_main_rgb: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            0_ultrawide_rgb: ...
        }
        """
        if out is None:
            out = dict()
        for camera_name, camera in self.cameras.items():
            this_out = None
            if camera_name in out:
                this_out = out[camera_name]
            this_out = camera.get(k=k, out=this_out)
            out[camera_name] = this_out
        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera_name in enumerate(self.cameras):
            camera = self.cameras[camera_name]
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i : i + 1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
        return out

    def start_recording(self, video_path: str, start_time: float):
        # directory
        video_dir = pathlib.Path(video_path)
        assert video_dir.parent.is_dir()
        video_dir.mkdir(parents=True, exist_ok=True)

        for camera_key, camera in self.cameras.items():
            cur_video_dir = str(video_dir.joinpath(f"{camera_key}.mp4").absolute())
            camera.start_recording(cur_video_dir, start_time)

    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()

    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [copy.deepcopy(x) for _ in range(n)]
    assert len(x) == n
    return x
