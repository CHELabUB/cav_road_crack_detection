# This code is written for tunning the camera yaw pitch and roll.
# Since camera mount limited the rolling angle, yaw and pitch will be the main parameters to be tunned.
# Written with the help of Copilot.
# Author: Haosong Xiao
# Date: 2025-09-23

# Example usage: python3 offline_tunning.py --folder Test_20260610_002 --image frame_1781127357377_50.png --model models/best.pt
# just change the folder and image names after a pipeline run with checkerboard.
# Please note that the checkerboard can also change to stop sign with known hight (known height for depth)

# library imports
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from utilities.pixel_conversion import Cameras_HX
from utilities.YOLO_gt_HX import Yolo_gt_HX


script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'utilities', 'config', 'camera.json')

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='Test_20260528_005', help='Data folder name under data/')
parser.add_argument('--image', default='frame_1779992739373_1.png', help='Image filename')
parser.add_argument('--model', default=os.path.join(script_dir, 'utilities', 'best.pt'), help='Path to YOLO model file')
args = parser.parse_args()

target_image_path = os.path.join(script_dir, 'data', args.folder, 'image_folder', args.image)
sync_report_path = os.path.join(script_dir, 'data', args.folder, 'sync_report_log.txt')
crop_log_path = os.path.join(script_dir, 'data', args.folder, 'detection_record_1_log.txt')
crack_location_path = os.path.join(script_dir, 'data', args.folder, 'camera_parameters.txt')


def get_box(width, height, u, v, box_size=512):

    half_box = box_size // 2
    x1 = u - half_box
    x2 = u + half_box
    y1 = v - half_box
    y2 = v + half_box

    if x1 < 0:
        x1 = 0
        x2 = box_size
    elif x2 > width:
        x2 = width
        x1 = width - box_size
    if y1 < 0:
        y1 = 0
        y2 = box_size
    elif y2 > height:
        y2 = height
        y1 = height - box_size

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return int(x1), int(x2), int(y1), int(y2)


def edge_projection(crop_center, Yolo_center, width, height):

    u, v = crop_center
    u_yolo, v_yolo = Yolo_center
    dx = u_yolo - u
    dy = v_yolo - v

    if dx == 0:
        edge_x = u
        edge_y = v + (height / 2) * (1 if dy > 0 else -1)
    elif dy == 0:
        edge_x = u + (width / 2) * (1 if dx > 0 else -1)
        edge_y = v
    else:
        scale_x = (width / 2) / abs(dx)
        scale_y = (height / 2) / abs(dy)
        scale = min(scale_x, scale_y)
        edge_x = u + dx * scale
        edge_y = v + dy * scale

    return (edge_x, edge_y)


def finding_the_scores(YOLO_box, CROP_box, edge_coord, box_size=512):
    x1, y1, x2, y2 = CROP_box
    x1_yolo, y1_yolo, x2_yolo, y2_yolo = YOLO_box
    R = np.sqrt((box_size / 2)**2 + (box_size / 2)**2)

    # center points
    u = (x1 + x2) / 2
    v = (y1 + y2) / 2
    u_yolo = (x1_yolo + x2_yolo) / 2
    v_yolo = (y1_yolo + y2_yolo) / 2

    # ROI: check the ratio between overlapped ground truth (gt, YOLO box) and whole gt.
    x1_inter = max(0, min(x2, x2_yolo) - max(x1, x1_yolo))
    y1_inter = max(0, min(y2, y2_yolo) - max(y1, y1_yolo))
    intersection_area = x1_inter * y1_inter
    yolo_area = (x2_yolo - x1_yolo) * (y2_yolo - y1_yolo)
    overlapping_rate = intersection_area / yolo_area if yolo_area > 0 else 0

    # AOT: check how close the YOLO center to the CROP center
    u_edge, v_edge = edge_coord
    if u_edge == u and v_edge == v:
        alignment_rate = 1.0  # Perfect alignment if edge coincides with crop center
    elif u_yolo < x1 or u_yolo > x2 or v_yolo < y1 or v_yolo > y2:
        alignment_rate = 0.0  # No alignment if YOLO center is outside the crop box
    else:
        # compute the length from edge to yolo center
        edge_to_yolo = np.sqrt((u_edge - u_yolo)**2 + (v_edge - v_yolo)**2)
        # compute the length from edge to crop center
        edge_to_crop = np.sqrt((u_edge - u)**2 + (v_edge - v)**2)

        # compute the length from yolo center to cropping center
        # please note that if edge to crop is less than 0, we consider it as out of bound.
        yolo_to_crop = np.sqrt((u_yolo - u)**2 + (v_yolo - v)**2)
        alignment_rate = (R - yolo_to_crop) / R if edge_to_crop > 0 else 0

    print("overlapping_rate, alignment_rate:", (overlapping_rate, alignment_rate))
    overall_score = (overlapping_rate + alignment_rate) / 2 # normalized to 0-1, raw is also fine.
    return overall_score


def read_crack_location(crack_location_path):

    crack_lat = None
    crack_lon = None

    with open(crack_location_path, 'r') as f:
        content = f.read()
        for param in content.split(','):
            param = param.strip()
            if param.startswith('crack_lat:'):
                crack_lat = float(param.split(':')[1].strip())
            elif param.startswith('crack_lon:'):
                crack_lon = float(param.split(':')[1].strip())

    return crack_lat, crack_lon


def extract_timestamp_from_image(image_path):

    filename = os.path.basename(image_path)
    parts = filename.replace('.png', '').split('_')
    if len(parts) >= 2:
        timestamp = int(parts[1])
        return timestamp


def find_synced_crop_time(sync_report_path, image_timestamp):

    with open(sync_report_path, 'r') as f:
        for line in f:
            if 'synced img time:' in line:
                parts = line.strip().split(',')
                synced_img_time = int(parts[2].split(':')[1].strip())
                synced_crop_time = int(parts[1].split(':')[1].strip())

                if synced_img_time == image_timestamp:
                    return synced_crop_time


def find_car_data_from_detection(crop_log_path, synced_crop_time):

    with open(crop_log_path, 'r') as f:
        for line in f:
            if 'RTK gen_time:' in line:
                # Extract RTK gen_time
                parts = line.strip().split(',')
                rtk_gen_time = int(parts[2].split(':')[1].strip())

                # Match with synced crop time
                if rtk_gen_time == synced_crop_time:
                    car_lat = float(parts[5].split(':')[1].strip())
                    car_lon = float(parts[6].split(':')[1].strip())
                    heading = float(parts[7].split(':')[1].strip())
                    return car_lat, car_lon, heading


if __name__ == "__main__":

    # Read crack location from file
    crack_lat = 43.002743501058376
    crack_long = -78.786735395743
    print(f"Loaded crack location - Lat: {crack_lat}, Lon: {crack_long}")

    # Extract timestamp from target image
    image_timestamp = extract_timestamp_from_image(target_image_path)
    print(f"Image timestamp: {image_timestamp}")

    # Find synced crop time from sync report
    synced_crop_time = find_synced_crop_time(sync_report_path, image_timestamp)
    print(f"Synced crop time: {synced_crop_time}")

    # Find car data from detection record (synced_crop_time matches RTK gen_time)
    car_lat, car_long, car_heading = find_car_data_from_detection(crop_log_path, synced_crop_time)
    print(f"Car position - Lat: {car_lat}, Lon: {car_long}, Heading: {car_heading}")

    # Find the YOLO box coordinates
    YOLO_box = Yolo_gt_HX(target_image_path, confidence_threshold=0.5, model_path=args.model)[0]

    # whole process finding score for each pitch and yaw combination.

    def tune_pitch_yaw(pitch, yaw):
        model = Cameras_HX(config_path, yaw, pitch)
        width = model.image_width
        height = model.image_height
        YOLO_center = ((YOLO_box[0] + YOLO_box[2]) / 2, (YOLO_box[1] + YOLO_box[3]) / 2)

        # logged -> 3D to 2D pixel conversion -> crop box
        x_enu, y_enu = model.gps_to_local(crack_lat, crack_long, car_lat, car_long)
        crack_car = model.enu_to_car_frame([x_enu, y_enu], car_heading)
        crack_camera = model.vehicle_to_camera_frame(crack_car)
        u, v = model.project_to_pixel(crack_camera)
        x1, x2, y1, y2 = get_box(width, height, u, v, box_size=512)

        # computing scores
        CROP_box = (x1, y1, x2, y2)
        CROP_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        edge_coord = edge_projection(CROP_center, YOLO_center, 512, 512)
        score = finding_the_scores(YOLO_box, CROP_box, edge_coord)

        return score

    # Define the range for pitch and yaw
    pitch_range = np.arange(-20, 11, 1)
    yaw_range = np.arange(-10, 11, 1)

    score_matrix = np.zeros((len(pitch_range), len(yaw_range)))
    # Brute-force search
    for i, pitch in enumerate(pitch_range):
        for j, yaw in enumerate(yaw_range):
            print(f"Testing pitch: {pitch}, yaw: {yaw}")
            score_matrix[i, j] = tune_pitch_yaw(pitch, yaw)

    max_score = np.max(score_matrix)
    max_indices = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)
    best_pitch = pitch_range[max_indices[0]]
    best_yaw = yaw_range[max_indices[1]]

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(score_matrix, cmap='viridis', interpolation='nearest', origin='lower',
                    extent=[yaw_range.min(), yaw_range.max(), pitch_range.min(), pitch_range.max()])
    fig.colorbar(cax, label='Score')
    ax.set_xlabel('Yaw (degrees)')
    ax.set_ylabel('Pitch (degrees)')
    ax.set_title('Score Heatmap')
    ax.text(best_yaw, best_pitch, f'Best\nScore: {max_score:.2f}',
            ha='center', va='center', color='white', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.5))

    print(f"Best score: {max_score:.2f} at pitch: {best_pitch}, yaw: {best_yaw}")

    plt.savefig('calibration_heatmap.png')
    plt.show()
