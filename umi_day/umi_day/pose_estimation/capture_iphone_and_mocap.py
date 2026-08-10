"""Log iPhone pose (iPhone connected via ethernet to computer) and mocap pose to a file."""

from flask import Flask
from flask_socketio import SocketIO
import base64
import numpy as np
import struct
from scipy.spatial.transform import Rotation
from umi_day.pose_estimation.util.pose_viewer import PoseViewer
import threading
from argparse import ArgumentParser
from dataclasses import dataclass
from util.mocap.mocap_node import MocapNode
from typing import Optional
import os
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class TimeStampedPose:
    # https://developer.apple.com/documentation/arkit/arframe/2867973-timestamp
    # https://stackoverflow.com/questions/45320278/arkit-what-does-the-arframe-timestamp-represent
    def __init__(self, transform_matrix: np.ndarray, timestamp: float):
        self.transform_matrix = transform_matrix.copy()
        # this only represents the uptime of the iphone
        self.timestamp = timestamp

    @property
    def position(self):
        return self.transform_matrix[:3, 3]
    
    @property
    def rotation(self):
        return Rotation.from_matrix(self.transform_matrix[:3, :3]).as_euler('XYZ', degrees=True)

    def __str__(self):
        return f"Translation: {self.position}, Rotation: {self.rotation}, Timestamp: {self.timestamp:.3f}"

def decode_data(encoded_str):
    # Decode the base64 string to bytes
    data_bytes = base64.b64decode(encoded_str)

    transform_matrix = np.zeros((4, 4))
    # Unpack transform matrix (16 floats)
    for i in range(4):
        for j in range(4):
            transform_matrix[i, j] = struct.unpack(
                "f", data_bytes[4 * (4 * i + j) : 4 * (4 * i + j + 1)]
            )[0]
    # The transform matrix is stored in column-major order in swift, so we need to transpose it in python
    transform_matrix = transform_matrix.T

    # Unpack timestamp (1 double)
    timestamp = struct.unpack("d", data_bytes[64:72])[0]

    return TimeStampedPose(transform_matrix, timestamp)

app = Flask(__name__)
socketio = SocketIO(app)

@dataclass
class MocapParams():
    left_rigid_body_id: int
    right_rigid_body_id: int
    ip: str

class ServerState:
    def __init__(self, record_duration=400, sides=['left', 'right'], use_mocap=False, mocap_params: Optional[MocapParams]=None, run_name: str='unnamed'):
        self.sides = sides
        self.record_mocap = use_mocap
        self.run_name = run_name

        self.cur_poses = {side: TimeStampedPose(np.eye(4), 1) for side in self.sides}
        self.lock = threading.Lock()

        self.pose_viewer = PoseViewer(len(sides))

        # recording
        self.record_duration = record_duration
        self.do_recording = record_duration > 0
        if self.do_recording:
            self.record_timestep = 0
            if self.record_mocap:
                assert mocap_params is not None
                self.mocap_params = mocap_params

                rigid_body_dict = {}
                if 'left' in self.sides: 
                    rigid_body_dict[mocap_params.left_rigid_body_id] = "iphone_left"
                if 'right' in self.sides:
                    rigid_body_dict[mocap_params.right_rigid_body_id] = "iphone_right"

                self.mocap_agent = MocapNode(rigid_body_dict=rigid_body_dict, ip=mocap_params.ip)

                # wait for mocap to connect
                if not all(self.mocap_agent.is_ready.values()):
                    print('waiting for all mocap rigid bodies to become detected before proceeding...')
                while not all(self.mocap_agent.is_ready.values()):
                    pass
                print('found all mocap rigid bodes!')

                self.mocap_poses = {side: np.zeros((self.record_duration,7)) for side in self.sides}
                self.mocap_timestamps = {side: np.zeros(self.record_duration) for side in self.sides}
            self.iphone_poses = {side: np.zeros((self.record_duration,7)) for side in self.sides}
            self.iphone_timestamps = {side: np.zeros(self.record_duration) for side in self.sides}
        self.has_updated = {side: False for side in self.sides}

    def compute_pose_fps(self, old_pose, new_pose):
        return 1 / (new_pose.timestamp - old_pose.timestamp + 1e-8)

    def pose_update(self, side_name, pose):
        """side_name: 'left' or 'right' """
        if self.pose_viewer.finished:
            return False
        
        if side_name not in self.sides:
            return True
        
        self.lock.acquire()
        self.has_updated[side_name] = True

        # save the pose
        self.cur_poses[side_name] = pose

        # record the pose
        all_poses_have_updated = all(self.has_updated.values())
        if self.do_recording and self.pose_viewer.recording_start and all_poses_have_updated:
            if self.record_timestep == 0:
                print('Recording started')
            if self.record_timestep % 10 == 0:
                print(f'Recording timestep {self.record_timestep}')

            for side in self.sides:
                self.has_updated[side] = False

                # mocap
                if self.record_mocap:
                    mocap_rigid_body_id = self.mocap_params.left_rigid_body_id if side == 'left' else self.mocap_params.right_rigid_body_id
                    self.mocap_agent.lock.acquire()
                    self.mocap_poses[side][self.record_timestep,:3] = self.mocap_agent.trans[mocap_rigid_body_id]
                    self.mocap_poses[side][self.record_timestep,3:] = self.mocap_agent.quat_wxyz[mocap_rigid_body_id]
                    self.mocap_timestamps[side][self.record_timestep] = self.mocap_agent.time[mocap_rigid_body_id]
                    self.mocap_agent.lock.release()

                # iphone
                iphone_mat = self.cur_poses[side].transform_matrix
                self.iphone_poses[side][self.record_timestep,:3] = iphone_mat[:3, 3]
                self.iphone_poses[side][self.record_timestep,3:] = Rotation.from_matrix(iphone_mat[:3, :3]).as_quat(canonical=False, scalar_first=True)
                self.iphone_timestamps[side][self.record_timestep] = self.cur_poses[side].timestamp
            
            self.record_timestep += 1

            if self.record_timestep == self.record_duration:
                self.pose_viewer.recording_start = False
                self.record_timestep = 0
                print('Recording finished')

                result = {
                    'iphone': {
                        side: {
                            'pose': self.iphone_poses[side],
                            'time': self.iphone_timestamps[side]
                        } for side in self.sides
                    }
                }

                if self.record_mocap:
                     result['mocap'] = {
                         side: {
                            'pose': self.mocap_poses[side],
                            'time': self.mocap_timestamps[side]
                         } for side in self.sides
                     }

                fname = f'recorded_trajectories/{self.run_name}_iphone{f"_mocap" if self.record_mocap else ""}.pkl'
                np.save(open(fname, 'wb'), result)

                print(f'Wrote to {fname}')

        # visualize the pose
        self.pose_viewer.on_pose_update([self.cur_poses[side].transform_matrix for side in self.sides])

        self.lock.release()

        return True

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('updateLeft')
def handle_left_message(data):
    if not server_state.pose_update('left', decode_data(data)):
        os._exit(0)

@socketio.on('updateRight')
def handle_right_message(data):
    if not server_state.pose_update('right', decode_data(data)):
        os._exit(0)

@socketio.on('kill')
def handle_kill_message(data):
    socketio.stop()

def run_server_thread(socket_ip):
    socketio.run(app, host=socket_ip, port=5555)

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    parser = ArgumentParser()
    parser.add_argument('--record_duration', type=int, default=400, help='set to negative or zero value to disable recording')
    parser.add_argument('--sides', nargs='+', default=['left', 'right'])
    parser.add_argument('--record_mocap', action='store_true')
    parser.add_argument('--left_iphone_mocap_id', default=6, type=int)
    parser.add_argument('--right_iphone_mocap_id', default=7, type=int)
    parser.add_argument('--socket_ip', default='192.168.123.18')
    parser.add_argument('--mocap_ip', type=str, default='')
    parser.add_argument('--run_name', default='unnamed')
    args = parser.parse_args()

    mocap_params = MocapParams(left_rigid_body_id=args.left_iphone_mocap_id, right_rigid_body_id=args.right_iphone_mocap_id, ip=args.mocap_ip)

    server_state = ServerState(args.record_duration, args.sides, args.record_mocap, mocap_params, args.run_name)
    
    # socket server needs in separate thread so that it doesn't block the visualization thread
    server_thread = threading.Thread(target=run_server_thread, args=(args.socket_ip,))
    server_thread.start()

    print('Waiting for iPhone to connect socket before opening visualizer...')
    
    server_state.pose_viewer.start_processing_stream()    
    exit()
