# detection node written to sync crop info with image, mask them and send detection results, with help of copilot
# author: Haosong Xiao
# testing date: 11/02/2025

# native imports
import json
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# pip installs
import cv2
import numpy as np
import torch
import torch.nn as nn
# ros2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# from sensor_msgs.msg import Image
from std_msgs.msg import Int64MultiArray
# self written models
from utilities.pixel_conversion import Cameras_HX

# For loading LECSFormer
base_dir = os.path.dirname(os.path.abspath(__file__))
LECSFormer_dir = os.path.join(base_dir, '..', 'LECSFormer') # change pth directory, this is only on vehicle docker
sys.path.insert(0, LECSFormer_dir)
from config import _C
from networks.LECSFormer import LECSFormer

# For DeepCrack, inference path mounted in vehicle docker already
# sys.path.append("/inference/Deep_crack/")
# from model.deepcrack import DeepCrack as DetectionModel
# from trainer import DeepCrackTrainer as DetectionModelTrainer

class Detection_node(Node):
    def __init__(self, weights_path='/inference/Deep_crack/DeepCrack_CT260_FT1.pth',
                 model_type='deepcrack'):
        super().__init__('detection_node')

        init_time = self.get_clock().now()
        init_sec, init_nsec = init_time.seconds_nanoseconds()
        self.node_start_time = int((init_sec + init_nsec * 1e-9) * 1000)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.model_type = model_type.lower()
        self.weights_path = weights_path

        if self.model_type == 'lecsformer':
            self.get_logger().info(f'Loading LECSFormer model from: {weights_path}')
            self.model = self.load_lecsformer_model(weights_path)
        else:
            self.get_logger().info(f'Loading DeepCrack model from: {weights_path}')
            self.model = self.load_model()

        self.get_logger().info('Model loaded successfully')

        self.declare_parameter('normalization_mean', [114.0, 121.0, 134.0])

        # Subscribers
        self.test_num_sub = self.create_subscription(
            String, '/test_number', self.test_num_callback, 50)
        self.process_sub = self.create_subscription(
            Int64MultiArray, '/ready_to_process', self.process_callback, 50)

        self.ready_process = 0
        self.process_finished = 0
        self.results = []

        # callback
        self.create_timer(0.1, self.results_callback)
        self.get_logger().info('Waiting for process indicator to be ready...')
        self.result = self.create_publisher(String, '/detection_results', 10)
        self.finished_pub = self.create_publisher(String, '/process_finished', 10)

    def test_num_callback(self, msg: String):
        self.test_num = msg.data
        self.setup_test_folder()

    def setup_test_folder(self):
        if self.test_num is None:
            return
        base = os.path.dirname(os.path.realpath(__file__))
        test_str = f"Test_{self.test_num}"
        self.test_base_folder = os.path.join(base, '..', 'data', test_str)
        self.image_folder = os.path.join(self.test_base_folder, 'image_folder')
        self.detection_folder = os.path.join(self.test_base_folder, 'detection_folder')
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.test_base_folder, exist_ok=True)
        os.makedirs(self.detection_folder, exist_ok=True)
        os.chmod(self.image_folder, 0o777)
        os.chmod(self.test_base_folder, 0o777)
        os.chmod(self.detection_folder, 0o777)

    def process_callback(self, msg: Int64MultiArray):
        if len(msg.data) >= 3:
            self.ready_process = msg.data[2]
            self.recording_timestamp = msg.data[0]
            status = "started" if self.ready_process == 1 else "stopped"
            # self.get_logger().info(f'Recording {status}')
        else:
            self.get_logger().warn(f'wrong msg: {msg.data}, check what vehicle node is publishing')

    def load_model(self):
        model = DetectionModel().to(self.device)
        trainer = DetectionModelTrainer(model).to(self.device)

        try:
            state_dict = trainer.saver.load(self.weights_path, multi_gpu=False)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise
        return model

    def load_lecsformer_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.get_logger().info(f"Checkpoint type: {type(checkpoint)}")

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # load model weights
            self.get_logger().info("Loading model weights from state_dict...")
            config = _C.clone()
            # manually set for now
            config.image_size = [512, 512]
            config.num_classes = 1
            model = LECSFormer(img_size=config.image_size,
                               patch_size=config.MODEL.LECSFormer.PATCH_SIZE,
                               in_channels=config.MODEL.LECSFormer.IN_CHANS,
                               num_classes=config.num_classes,
                               embed_dim=config.MODEL.LECSFormer.EMBED_DIM,
                               depths=config.MODEL.LECSFormer.DEPTHS,
                               num_heads=config.MODEL.LECSFormer.NUM_HEADS,
                               window_size=config.MODEL.LECSFormer.WINDOW_SIZE,
                               mlp_ratio=config.MODEL.LECSFormer.MLP_RATIO,
                               qkv_bias=config.MODEL.LECSFormer.QKV_BIAS,
                               qk_scale=config.MODEL.LECSFormer.QK_SCALE,
                               drop_rate=config.MODEL.DROP_RATE,
                               drop_path_rate=config.MODEL.DROP_PATH_RATE,
                               patch_norm=config.MODEL.LECSFormer.PATCH_NORM,
                               use_checkpoint=config.TRAIN.USE_CHECKPOINT
                               ).cuda() # please note that our vehicle has cuda, change here if you don't have one on the vehicle
            try:
                model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except Exception as e:
                self.get_logger().info(f"Error loading model state_dict, try loading with DataParallel")
                model = nn.DataParallel(model)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    self.get_logger().info(f"Model loaded with DataParallel: {msg}")
                else:
                    msg = model.load_state_dict(checkpoint, strict=True)
                    self.get_logger().info(f"Model loaded with DataParallel: {msg}")
        else:
            self.get_logger().info("Loading full model...")
            model = checkpoint

        model.to(self.device)
        model.eval()
        self.get_logger().info(f"Model loaded successfully from {checkpoint_path}.")
        return model

    @staticmethod
    def np2Tensor(array):
        """Convert numpy array to tensor"""
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        elif array.ndim == 3:
            array = array.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")
        return torch.from_numpy(array).float()

    def get_mask(self, image):
        if self.model_type == 'lecsformer':
            # LECSFormer inference
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                norm_mean = self.get_parameter('normalization_mean').value
                image_norm = (image_rgb.astype(np.float32) - norm_mean) / 255.0
                input_tensor = self.np2Tensor(image_norm).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out, _ = self.model(input_tensor)
                    output = torch.sigmoid(out).squeeze(0)
                output_image = output.detach().cpu().numpy().squeeze(0) * 255
                mask = output_image.astype('uint8')
                return mask
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                self.get_logger().error(f'Error during LECSFormer inference: {str(e)}')
                raise
        else:
            # DeepCrack inference
            mean = np.array(self.get_parameter('normalization_mean').value,
                            dtype=np.float32)
            img_norm = (image - mean) / 255.0
            input_tensor = self.np2Tensor(img_norm).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(input_tensor)[0]
                pred = torch.sigmoid(pred).cpu().squeeze().numpy() * 255
            mask = pred.astype(np.uint8)
            return mask

    def compute_crack_data(self, mask, crop_offset_x, crop_offset_y, config_path=None, camera_yaw=0, camera_pitch=0):
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, '..', 'utilities', 'config', 'camera.json')
            yaw_pitch_path = os.path.join(script_dir, '..', 'utilities', 'config', 'tunable_params.json')
            with open(yaw_pitch_path, 'r') as f:
                tunable = json.load(f)
            camera_yaw = tunable.get('camera_yaw', camera_yaw)
            camera_pitch = tunable.get('camera_pitch', camera_pitch)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        orig_h = mask.shape[0]

        center_row = orig_h // 2
        row_start = center_row - 128
        row_end = center_row + 70
        mask_center = mask[row_start:row_end, :]
        h, w = mask_center.shape
        adjusted_offset_y = crop_offset_y + row_start # reconstruct to the original image pixel
        full_mask = np.zeros((1544, 2064), dtype=np.uint8)
        full_mask[adjusted_offset_y:adjusted_offset_y + h, crop_offset_x:crop_offset_x + w] = mask_center

        # reconstruct highlighted pixels back to 3D coordinates
        white_pixels = np.argwhere(full_mask > 0)
        if len(white_pixels) == 0:
            return None
        pixels_xy = white_pixels[:, [1, 0]]
        camera_model = Cameras_HX(config_path, camera_yaw, camera_pitch)
        real_coords = []
        valid_indices = []
        for i, pixel in enumerate(pixels_xy):
            pt = camera_model.pixel_to_vehicle_frame(tuple(pixel))
            if pt is not None:
                real_coords.append([pt[0], pt[1]])
                valid_indices.append(i)

        if not real_coords:
            return None

        real_coords = np.array(real_coords)
        pixels_xy = pixels_xy[valid_indices]
        center_real = np.array([
            real_coords[:, 0].min() + (real_coords[:, 0].max() - real_coords[:, 0].min()) / 2,
            real_coords[:, 1].min() + (real_coords[:, 1].max() - real_coords[:, 1].min()) / 2,
        ])
        top = real_coords[:, 0] > center_real[0]
        bottom = real_coords[:, 0] <= center_real[0]
        left = real_coords[:, 1] > center_real[1]
        right = real_coords[:, 1] <= center_real[1]

        corners_pixel = []
        corners_real = []
        for quadrant in [top & left, top & right, bottom & right, bottom & left]:
            idx = np.where(quadrant)[0]
            if len(idx) > 0:
                dists = np.sum((real_coords[idx] - center_real) ** 2, axis=1)
                best = idx[np.argmax(dists)]
                corners_pixel.append(tuple(pixels_xy[best]))
                corners_real.append(real_coords[best])
            else:
                corners_pixel.append(None)
                corners_real.append(None)

        if corners_real[0] is None or corners_real[2] is None:
            return None

        length_m = float(np.linalg.norm(corners_real[0] - corners_real[2]))
        return length_m, corners_pixel, row_start, row_end

    def publish_results(self):
        if not self.results:
            self.get_logger().info('No results to publish.')
            return

        msg = String()
        msg.data = str(self.results)
        self.result.publish(msg)

        finished_msg = String()
        finished_msg.data = str(self.process_finished)
        self.finished_pub.publish(finished_msg)

        self.get_logger().info(f'Published {self.results}')

    def process_all(self):
        # self.get_logger().info(f'ready to process: {self.ready_process}')
        if self.ready_process != 1:
            self.get_logger().info('Waiting for ready_process signal...')
            return

        crop_log = os.path.join(self.test_base_folder, 'detection_record_1_log.txt')
        raw_img_path = self.image_folder
        output_path = self.detection_folder

        self.get_logger().info('Loading cropping coords')

        detection_records = []
        with open(crop_log, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = re.search(r'record: (\d+),', line)
                if record and record.group(1) == '1':
                    timestamp_match = re.search(r'RTK gen_time:\s*(\d+)', line)
                    x1_match = re.search(r'x1:\s*([-\d]+)', line)
                    y1_match = re.search(r'y1:\s*([-\d]+)', line)
                    x2_match = re.search(r'x2:\s*([-\d]+)', line)
                    y2_match = re.search(r'y2:\s*([-\d]+)', line)
                    detection_records.append({
                        "timestamp": int(timestamp_match.group(1)) if timestamp_match else -1,
                        "x1": int(x1_match.group(1)) if x1_match else -1,
                        "y1": int(y1_match.group(1)) if y1_match else -1,
                        "x2": int(x2_match.group(1)) if x2_match else -1,
                        "y2": int(y2_match.group(1)) if y2_match else -1
                    })
        if not detection_records:
            self.get_logger().warn("No detection records found.")
            return

        self.get_logger().info(f'Loaded {len(detection_records)} croppable images')

        # Create output folders
        os.makedirs(output_path, exist_ok=True)
        crop_folder = os.path.join(output_path, 'crop')
        annotate_folder = os.path.join(output_path, 'annotate')
        mask_folder = os.path.join(output_path, 'mask')
        os.makedirs(crop_folder, exist_ok=True)
        os.makedirs(annotate_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        # SYNC images to the crop record
        image_files = sorted([f for f in os.listdir(raw_img_path) if f.endswith('.png')])
        if not image_files:
            self.get_logger().warn("No images found in image folder.")
            return

        self.get_logger().info(f'Found {len(image_files)} croppable images to process')

        processed_count = 0
        for img_file in image_files:
            match = re.match(r'frame_(\d+)_(\d+)\.png', img_file)
            if not match:
                self.get_logger().warn(f"wrong frame name? {img_file}, check the node savings")
                continue

            img_ts = int(match.group(1))
            frame_count = int(match.group(2))
            raw_path = os.path.join(raw_img_path, img_file)
            raw_img = cv2.imread(raw_path)
            if raw_img is None:
                continue

            # Find closest detection record by timestamp
            ts_diff = [abs(img_ts - det["timestamp"]) for det in detection_records]
            min_idx = ts_diff.index(min(ts_diff))
            det_info = detection_records[min_idx]

            x1, y1, x2, y2 = det_info["x1"], det_info["y1"], det_info["x2"], det_info["y2"]
            annotated_img = raw_img.copy()

            # log synced results:
            log_msg = f"Node start time: {self.node_start_time}, \
            synced crop time: {det_info['timestamp']}, \
            synced img time: {img_ts}\n"

            sync_record_log = os.path.join(self.test_base_folder, 'sync_report_log.txt')
            with open(sync_record_log, 'a') as f:
                f.write(log_msg)

            if all(v != -1 for v in [x1, y1, x2, y2]):
                cropped_img = raw_img[y1:y2, x1:x2]
                crop_path = os.path.join(crop_folder, f'cropped_{img_ts}_{frame_count}.png')
                if not cv2.imwrite(crop_path, cropped_img):
                    self.get_logger().warn(f"Failed to save cropped image: {crop_path}")
                    continue

                mask = self.get_mask(cropped_img)
                mask_path = os.path.join(mask_folder, f'mask_{img_ts}_{frame_count}.png')
                cv2.imwrite(mask_path, mask)

                # Calculate crack length and draw corners on annotation
                now_computed = self.get_clock().now().seconds_nanoseconds()
                computed_timestamp = int((now_computed[0] + now_computed[1] * 1e-9) * 1000)
                crack_data = self.compute_crack_data(mask, x1, y1)
                crack_length = crack_data[0] if crack_data is not None else None
                if crack_length is not None:
                    self.get_logger().info(f'Crack length for {img_file}: {crack_length:.4f} meters')
                    crack_log = os.path.join(self.test_base_folder, 'crack_length_log.txt')
                    if not os.path.exists(crack_log) or os.path.getsize(crack_log) == 0:
                        with open(crack_log, 'w') as f:
                            f.write('image_name,computed_timestamp,synced_crop_timestamp,crack_length_m\n')
                    with open(crack_log, 'a') as f:
                        f.write(f"{img_file},{computed_timestamp},{det_info['timestamp']},{crack_length:.4f}\n")
                    self.results.append((computed_timestamp, crack_length))

                    # Draw detection band and TL/BR corner circles on annotation
                    _, corners_pixel, row_start, row_end = crack_data
                    cv2.rectangle(annotated_img, (x1, y1 + row_start), (x2, y1 + row_end), (0, 255, 0), 2)
                    for corner in [corners_pixel[0], corners_pixel[2]]:
                        if corner is not None:
                            cv2.circle(annotated_img, corner, 12, (255, 0, 0), 2)
                else:
                    self.get_logger().warn(f'Failed to calculate crack length for {img_file}')

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            annotated_path = os.path.join(annotate_folder, f'annotated_{img_ts}_{frame_count}.png')
            cv2.imwrite(annotated_path, annotated_img)
            processed_count += 1

            if processed_count % 10 == 0:
                self.get_logger().info(f'Processed {processed_count}/{len(image_files)} images')

        self.process_finished = 1

    def results_callback(self):
        self.get_logger().info(f'ready to process: {self.ready_process}')
        if self.ready_process == 1 and self.process_finished == 0:
            self.process_all()
        elif self.ready_process == 1 and self.process_finished == 1:
            self.publish_results()
        elif self.ready_process != 1 and self.process_finished == 1:
            self.process_finished = 0


def main(args=None):
    rclpy.init(args=args)
    node = None
    current_dir = os.getcwd()
    model_pth = os.path.join(current_dir, 'models', 'combined_260_111_332_aug_v3.pth')
    try:
        node = Detection_node(weights_path=model_pth, model_type='lecsformer')
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info('Interrupted, shutting down.')
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
