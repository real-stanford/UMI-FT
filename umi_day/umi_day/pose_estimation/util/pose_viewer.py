import numpy as np

from threading import Event

import numpy as np
import open3d as o3d

from PIL import Image

from scipy.spatial.transform import Rotation

class PoseViewer:
    def __init__(self, num_poses):
        self.event = Event()
        self.init_camera_poses = None
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.register_key_callback(ord("S"), self.on_key_s)
        self.vis.register_key_callback(ord("C"), self.on_key_c)
        print(f'Visualizer opened: Press "s" to start recording. Press "c" to stop recording.')
        self.vis.create_window("iPhone Point Cloud Steaming", width=1024, height=1024)
        self.vis.get_view_control()
        self.poses = []
        self.num_poses = num_poses

        self.prev_pose_matrices = []
        self.camera_geoms = []
        self.finished = False
        self.recording_start = False

    def on_key_s(self, vis):
        self.recording_start = True

    def on_key_c(self, vis):
        self.finished = True

    def on_pose_update(self, poses):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        assert len(poses) == self.num_poses
        self.poses = poses

        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def create_wireframe_cube(self, size=1.0):
        points = [[-size, -size, -size],
                [-size, -size, size],
                [-size, size, -size],
                [-size, size, size],
                [size, -size, -size],
                [size, -size, size],
                [size, size, -size],
                [size, size, size]]

        lines = [[0, 1], [1, 3], [3, 2], [2, 0],
                [4, 5], [5, 7], [7, 6], [6, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        return line_set
       
    def start_processing_stream(self):
        prev_pose_matrices = []
        camera_geoms = []
        
        while True:
            self.event.wait()
        
            if self.init_camera_poses is None:
                self.init_camera_poses = self.poses

                # cube
                cube = self.create_wireframe_cube(size=0.6)
                self.vis.add_geometry(cube)
                
                # camera coordinate frame
                for cam_i in range(self.num_poses):
                    camera_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(camera_geom.transform(self.poses[cam_i]))
                    camera_geoms.append(camera_geom)
            else:
                # Update visualization
                for cam_i in range(self.num_poses):
                    extrinsic = self.poses[cam_i]

                    # Update camera coordinate frame
                    camera_geom = camera_geoms.pop(cam_i)
                    self.vis.remove_geometry(camera_geom, reset_bounding_box=False)
                    camera_geom = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    self.vis.add_geometry(camera_geom.transform(extrinsic), reset_bounding_box=False)
                    camera_geoms.insert(cam_i, camera_geom)

                if not self.vis.poll_events():
                    self.finished = True
                    break
                self.vis.update_renderer()

            prev_pose_matrices = self.poses
            self.event.clear()

            if self.finished:
                break
