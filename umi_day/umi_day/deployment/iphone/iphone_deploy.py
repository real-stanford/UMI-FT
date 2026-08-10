"""Log iPhone pose (iPhone connected via ethernet to computer) to a file."""

from flask import Flask
from flask_socketio import SocketIO
import base64
import numpy as np
import threading
import cv2
from argparse import ArgumentParser
from PIL import Image
from io import BytesIO
import logging
from threading import Event
from datetime import datetime
from multiprocessing import Process
from umi_day.common.qr_code_util import dynamic_qr_timecode, read_qr_code
from umi_day.common.cv_util import get_image_transform_with_border

def decode_rgb_base64_to_numpy(base64_string):
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)
    
    # Convert bytes into a PIL image
    image = Image.open(BytesIO(image_data))
    
    # Convert to NumPy array
    numpy_image = np.array(image)
    
    return numpy_image

def decode_depth_base64_to_numpy(base64_string, max_depth=1.0):
    # Decode Base64 string to bytes
    depth_bytes = base64.b64decode(base64_string)

    # Convert bytes to NumPy array of float32
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)

    # Reshape to original image size (height, width)
    depth_array = depth_array.reshape((240, 320))

    # Clip depth values to max_depth
    depth_array = np.clip(depth_array, 0, max_depth)

    # Scale depth values from [0, max_depth] to [0, 255]
    depth_array = (depth_array / max_depth * 255).astype(np.uint8)

    return depth_array

def decode_depth_base64_to_numpy_raw(base64_string, max_depth=0.5):
    # Decode Base64 string to bytes
    depth_bytes = base64.b64decode(base64_string)

    # Convert bytes to NumPy array of float32
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32)

    # Reshape to original image size (height, width)
    depth_array = depth_array.reshape((240, 320))

    # Clip depth values to max_depth
    depth_array = np.clip(depth_array, 0, max_depth)

    return depth_array

def compute_rgb_latency(frame, frame_received_time: datetime):
    qr_data = read_qr_code(frame)
    if qr_data:
        try:
            qr_time = datetime.fromisoformat(qr_data)
            deltaSeconds = (frame_received_time - qr_time).total_seconds()
            return deltaSeconds
        except:
            return None
    else:
        return None

class ServerState:
    def __init__(self, eval_latency):
        self.eval_latency = eval_latency
        self.frame = None
        self.event = Event()
        self.is_depth = False

    def set_frame(self, rgb, is_depth):
        self.frame = rgb
        self.is_depth = is_depth
        self.event.set()

        if self.eval_latency:
            received_time = datetime.now()
            latencySeconds = compute_rgb_latency(rgb, received_time)
            if latencySeconds:
                print(f'Frame latency: {latencySeconds:.3f} seconds')

    def get_blocking(self):
        self.event.wait()
        self.event.clear()
        return self.frame, self.is_depth

def visualize(server_states: list[ServerState], policy_obs: bool):
    """ Continuously updates the OpenCV window whenever a new image is received. """
    policy_obs_transform = get_image_transform_with_border(in_res=(320, 240), out_res=(224, 224), bgr_to_rgb=False)

    while True:
        # wait for all camera feeds to have new data
        ims = []
        for server_state in server_states:
            im, is_depth = server_state.get_blocking()

            if is_depth:
                im = np.repeat(im[:,:,np.newaxis], 3, axis=2) # convert depth image to rgb channels

            im = policy_obs_transform(im) if policy_obs else im
            ims.append(im)
        
        # display received images
        vis = np.hstack(ims)
        cv2.imshow("Camera Feed", vis[:,:,::-1])
    
        # Required to keep OpenCV window responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def run_server_on_thread(socket_ip: str, socket_port: int, eval_latency: bool, max_depth: float) -> ServerState:
    # Setup server state
    server_state = ServerState(eval_latency)

    # Supress logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    # Create socket and callbacks
    app = Flask(__name__)
    socketio = SocketIO(app)

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        pass

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
        pass

    @socketio.on('rgb')
    def handle_rgb_message(data):
        image_np = decode_rgb_base64_to_numpy(data)
        server_state.set_frame(image_np, is_depth=False)

    @socketio.on('depth')
    def handle_depth_message(data):
        image_np = decode_depth_base64_to_numpy_raw(data, max_depth=max_depth)
        server_state.set_frame(image_np, is_depth=True)

    @socketio.on('validate')
    def handle_validate_message(data):
        """This endpoint is for the client to be able to check if the server can acknowledge a mesage received."""
        return 'server got it!'

    @socketio.on('kill')
    def handle_kill_message(data):
        socketio.stop()

    def thread():
        socketio.run(app, host=socket_ip, port=socket_port, allow_unsafe_werkzeug=True) # TODO: remove allow_unsafe_werkzeug or check why it's not meant for production

    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=thread, args=(), daemon=False)
    server_thread.start()

    return server_state

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--socket_ip', default='192.168.123.18')
    parser.add_argument('--socket_ports', default=[5556, 5555, 5557], nargs='+', type=int) # 5556 is ultrawide, 5555 is main, 5557 is depth
    parser.add_argument('--eval_latency', action='store_true')
    parser.add_argument('--policy_obs', action='store_true', help='Pass this flag to resize the image to match the policy observation format')
    parser.add_argument('--max_depth', type=float, default=1.0, help='Maximum depth value in meters')
    args = parser.parse_args()

    # Start a separate process for QR code if eval_latency is enabled
    if args.eval_latency:
        qr_process = Process(target=dynamic_qr_timecode, daemon=False)
        qr_process.start()

    # Start each server on a separate thread
    server_states = []
    for port in args.socket_ports:
        server_states.append(run_server_on_thread(args.socket_ip, port, args.eval_latency, args.max_depth))

    # Start the OpenCV visualization on the main thread
    visualize(server_states, args.policy_obs)
    