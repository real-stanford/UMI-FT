# Copyright © 2018 Naturalpoint
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import threading
from typing import Dict, List, cast
import numpy as np
import sys
import time
from .natnet_client import NatNetClient
from . import mocap_data as MoCapData

from functools import partial

from transforms3d import quaternions, affines


def receive_new_frame(
    data_dict,
    rigid_body_dict: Dict[int, str]
):
    print(f"receive_new_frame: {data_dict}")
    model_names = []
    marker_data_list = data_dict["marker_set_data"].marker_data_list
    for marker_data in marker_data_list:
        model_name = marker_data.model_name.decode("utf-8")
        if model_name != "all":
            model_names.append(model_name)


    rigid_body_list = data_dict["rigid_body_data"].rigid_body_list
    rigid_body_list = cast(List[MoCapData.RigidBody], rigid_body_list)


    for i, rigid_body in enumerate(rigid_body_list):
        if rigid_body.id_num not in rigid_body_dict:
            continue
        rigid_body_name = rigid_body_dict[rigid_body.id_num]
        # name = rigid_body_name

        # time_now = self.get_clock().now().to_msg()
        mocap_robot_in_world_frame = affines.compose(
            T=rigid_body.pos,
            R=quaternions.quat2mat(
                np.array(rigid_body.rot)[[3, 0, 1, 2]]
            ),  # rigid_body.rot is xyzw, need to convert to wxyz
            Z=np.ones(3),
        )
            
        trans, rotm, _, _ = affines.decompose(mocap_robot_in_world_frame)
        quat_wxyz = quaternions.mat2quat(rotm)
        print(trans, quat_wxyz)

class MocapNode():
    def __init__(
        self,
        rigid_body_dict: Dict[int, str],
        ip: str,
        use_multicast=True,
    ):
        assert ip, 'server ip must be provided to initialize mocap'
        self.is_ready = {id: False for id in rigid_body_dict}
        self.trans = {id: None for id in rigid_body_dict}
        self.quat_wxyz = {id: None for id in rigid_body_dict}
        self.time = {id: None for id in rigid_body_dict}
    
        self.prev_receive_time = time.monotonic()

        self.lock = threading.Lock()

        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address("127.0.0.1")
        self.streaming_client.set_server_address(ip)
        self.streaming_client.set_use_multicast(use_multicast)

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        self.streaming_client.new_frame_listener = partial(
            self.receive_new_frame,
            rigid_body_dict = rigid_body_dict,
        )

        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        is_running = self.streaming_client.run()
        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting 1")

        time.sleep(1)
        if self.streaming_client.connected() is False:
            print(
                "ERROR: Could not connect properly.  Check that Motive streaming is on."
            )
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting 2")

        print("mocap init done")
        

    def receive_new_frame(
        self,
        data_dict,
        rigid_body_dict: Dict[int, str]
    ):
        # print(f"receive_new_frame: {data_dict}")
        model_names = []
        marker_data_list = data_dict["marker_set_data"].marker_data_list
        for marker_data in marker_data_list:
            model_name = marker_data.model_name.decode("utf-8")
            if model_name != "all":
                model_names.append(model_name)

        rigid_body_list = data_dict["rigid_body_data"].rigid_body_list
        rigid_body_list = cast(List[MoCapData.RigidBody], rigid_body_list)

        self.lock.acquire()
        for i, rigid_body in enumerate(rigid_body_list):
            if rigid_body.id_num not in rigid_body_dict:
                continue
            rigid_body_name = rigid_body_dict[rigid_body.id_num]
            # name = rigid_body_name

            # time_now = self.get_clock().now().to_msg()
            mocap_robot_in_world_frame = affines.compose(
                T=rigid_body.pos,
                R=quaternions.quat2mat(
                    np.array(rigid_body.rot)[[3, 0, 1, 2]]
                ),  # rigid_body.rot is xyzw, need to convert to wxyz
                Z=np.ones(3),
            )
                
            trans, rotm, _, _ = affines.decompose(mocap_robot_in_world_frame)
            quat_wxyz = quaternions.mat2quat(rotm)

            # print(trans, quat_wxyz,time.monotonic())

            self.is_ready[rigid_body.id_num] = True
            self.trans[rigid_body.id_num] = trans
            self.quat_wxyz[rigid_body.id_num] = quat_wxyz
            self.time[rigid_body.id_num] = data_dict['timestamp']
        self.lock.release()

        self.prev_receive_time = time.monotonic()

    def disconnect(self):
        self.streaming_client.shutdown()
