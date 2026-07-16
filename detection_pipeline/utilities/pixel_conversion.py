# code written for pixel to world conversion and vice versa.
# Author: Haosong Xiao

# native imports
import os
import json
# need to pip install
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config', 'camera.json')


class Cameras_HX:
    def __init__(self, config_dir, yaw=0, pitch=0):
        self.config = self.load_config(config_dir)
        self.camera_intrinsics = self.config['camera_intrinsic']
        self.image_height = self.camera_intrinsics['image_height']
        self.image_width = self.camera_intrinsics['image_width']
        self.camera_extrinsic = self.config['camera_extrinsic']
        self.camera_height = self.camera_extrinsic['camera_height']
        # self.camera_yaw = self.camera_extrinsic['camera_yaw']
        # self.camera_pitch = self.camera_extrinsic['camera_pitch']
        self.camera_yaw = yaw
        self.camera_pitch = pitch
        self.camera_roll = self.camera_extrinsic['camera_roll']

        # GPS to camera offset in vehicle frame (x: forward, y: left, z: up)
        self.gps_to_camera_offset_body = self.config['gps_to_camera_offset_body']
        self.offset_x = self.gps_to_camera_offset_body['forward']
        self.offset_y = self.gps_to_camera_offset_body['left']
        self.offset_z = self.camera_height

    def load_config(self, config_dir):
        with open(config_dir, 'r') as f:
            config = json.load(f)
        return config

    def gps_to_local(self, crack_lat, crack_long, car_lat, car_long):

        lat1, lon1 = np.radians(car_lat), np.radians(car_long)
        lat2, lon2 = np.radians(crack_lat), np.radians(crack_long)

        # earth radius has been pushed to config, for the sake of testing, I hard code it here
        R = 6371000

        # please note that the lat1 and long1 is the reference point
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        x = R * dlon * np.cos(lat1)  # East direction
        y = R * dlat                 # North direction

        return x, y

    def enu_to_car_frame(self, enu_point, car_heading):

        # GPS heading: 0° = North, 90° = East
        heading_rad = np.radians(car_heading)

        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)

        # this rotation gives crack w.r.t the car frame using car itself as origin
        R = np.array([[sin_h, cos_h],
                      [-cos_h, sin_h]])

        return R @ enu_point

    def vehicle_to_standard_camera_frame(self):

        # Mapping from vehicle to camera:
        # Vehicle X (Forward) → Camera Z (Forward)
        # Vehicle Y (Left) → Camera -X
        # Vehicle Z (Up) → Camera -Y

        R_vehicle_to_standard_camera = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        return R_vehicle_to_standard_camera

    def standard_camera_rotation_matrix(self):

        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)
        roll_rad = np.radians(self.camera_roll)

        cos_r = np.cos(roll_rad)
        sin_r = np.sin(roll_rad)
        cos_p = np.cos(pitch_rad)
        sin_p = np.sin(pitch_rad)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)

        R_pitch = np.array([[1, 0, 0],
                            [0, cos_p, -sin_p],
                            [0, sin_p, cos_p]])

        R_yaw = np.array([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]])

        R_roll = np.array([[cos_r, -sin_r, 0],
                           [sin_r, cos_r, 0],
                           [0, 0, 1]])

        R_zyx = R_roll @ R_pitch @ R_yaw
        return R_zyx

    def vehicle_to_camera_frame(self, crack_position_vehicle):

        if len(crack_position_vehicle) == 2:
            crack_3d = np.array([crack_position_vehicle[0], crack_position_vehicle[1], 0.0])
        else:
            crack_3d = np.array(crack_position_vehicle)
        crack_relative_to_camera = crack_3d - np.array([self.offset_x, self.offset_y, self.offset_z])
        # print("Crack relative to camera (Vehicle frame):", crack_relative_to_camera)

        R_vehicle_to_standard = self.vehicle_to_standard_camera_frame()
        crack_in_standard_camera = R_vehicle_to_standard @ crack_relative_to_camera
        R_camera_rotation = self.standard_camera_rotation_matrix()
        crack_in_camera_frame = R_camera_rotation @ crack_in_standard_camera

        return crack_in_camera_frame

    def project_to_pixel(self, point_3d_camera):

        X, Y, Z = point_3d_camera
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']

        if Z <= 0:
            return -1, -1

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        return int(u), int(v)

    def pixel_to_vehicle_frame(self, pixel_coords):

        u, v = pixel_coords
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']

        normalized_ray_camera = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

        R_camera_rotation = self.standard_camera_rotation_matrix()
        R_inv_camera_rotation = R_camera_rotation.T

        R_vehicle_to_standard = self.vehicle_to_standard_camera_frame()
        R_standard_to_vehicle = R_vehicle_to_standard.T

        ray_standard_camera = R_inv_camera_rotation @ normalized_ray_camera
        ray_vehicle = R_standard_to_vehicle @ ray_standard_camera

        cam_offset_vec = np.array([self.offset_x, self.offset_y, self.offset_z])
        if np.abs(ray_vehicle[2]) < 1e-6:
            return None

        d = -cam_offset_vec[2] / ray_vehicle[2]
        if d < 0:
            return None

        point_vehicle = cam_offset_vec + d * ray_vehicle

        return point_vehicle
