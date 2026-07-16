# code written to do offline length calculation of cracks in images from recorded data, with the help of copilot.
# please note that the camera parameters may change from different vehicle reservations and calibrations.
# author: Haosong Xiao

# native imports
import json
import os
import sys
# needs pip install
import cv2
import numpy as np

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base, 'detection_pipeline', 'utilities'))
from pixel_conversion import Cameras_HX


def compute_crack_data(mask_image, crop_offset_x, crop_offset_y, config_path, camera_yaw=0, camera_pitch=0):

    if mask_image.ndim == 3:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    orig_h, orig_w = mask.shape
    center_row = orig_h // 2
    row_start = center_row - 128
    row_end = center_row + 70
    mask_center = mask[row_start:row_end, :]
    h, w = mask_center.shape
    adjusted_offset_y = crop_offset_y + row_start

    # allocate the mask pixels in original images
    full_mask = np.zeros((1544, 2064), dtype=np.uint8)
    full_mask[adjusted_offset_y:adjusted_offset_y + h, crop_offset_x:crop_offset_x + w] = mask_center
    white_pixels = np.argwhere(full_mask > 0)
    if len(white_pixels) == 0:
        return None
    pixels_xy = white_pixels[:, [1, 0]]  # (row,col) -> (x,y)

    # back to 3D coordinate in the vehicle frame
    model = Cameras_HX(config_path, camera_yaw, camera_pitch)
    real_coords = []
    valid_indices = []
    for i, pixel in enumerate(pixels_xy):
        pt = model.pixel_to_vehicle_frame(tuple(pixel))
        if pt is not None:
            real_coords.append([pt[0], pt[1]])
            valid_indices.append(i)

    if not real_coords:
        return None

    real_coords = np.array(real_coords)
    pixels_xy = pixels_xy[valid_indices]

    x_min, x_max = real_coords[:, 0].min(), real_coords[:, 0].max()
    y_min, y_max = real_coords[:, 1].min(), real_coords[:, 1].max()
    center_real = np.array([
        x_min + (x_max - x_min) / 2,
        y_min + (y_max - y_min) / 2,
    ])

    # 4 quadrants
    top = real_coords[:, 0] > center_real[0]
    bottom = real_coords[:, 0] <= center_real[0]
    left = real_coords[:, 1] > center_real[1]
    right = real_coords[:, 1] <= center_real[1]
    corners_pixel = []
    corners_real = []
    for qmask in [top & left, top & right, bottom & right, bottom & left]:
        idx = np.where(qmask)[0]
        if len(idx) > 0:
            dists = np.sum((real_coords[idx] - center_real) ** 2, axis=1)
            best = idx[np.argmax(dists)]
            corners_pixel.append(tuple(pixels_xy[best]))
            corners_real.append(real_coords[best])
        else:
            corners_pixel.append(None)
            corners_real.append(None)

    # only consider TL to BR for length estimation
    if corners_real[0] is None or corners_real[2] is None:
        return None

    length_m = float(np.linalg.norm(corners_real[0] - corners_real[2]))
    return length_m, corners_pixel, row_start, row_end, orig_h, orig_w


def visualize_crack_length(mask_image, crop_offset_x, crop_offset_y,
                           config_path, original_image, output_path,
                           camera_yaw=0, camera_pitch=0):

    result = compute_crack_data(mask_image, crop_offset_x, crop_offset_y,
                                config_path, camera_yaw, camera_pitch)
    if result is None:
        return None

    length_m, corners_pixel, row_start, row_end, orig_h, orig_w = result

    # --- extract crop from original image (full 512×512) ---------------------
    img_crop = original_image[crop_offset_y:crop_offset_y + orig_h,
                              crop_offset_x:crop_offset_x + orig_w].copy()

    # --- draw white pixels directly on crop (single pixel, no thickening) ----
    if mask_image.ndim == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_image.copy()
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    all_mask_pixels = np.argwhere(mask_binary > 0)
    for pixel in all_mask_pixels:
        y, x = pixel
        if 0 <= y < orig_h and 0 <= x < orig_w:
            img_crop[y, x] = [255, 255, 255]

    # --- green rectangle for detection band ----------------------------------
    cv2.rectangle(img_crop,
                  (0, row_start), (orig_w, row_end),
                  (0, 255, 0), 3)

    # --- blue hollow circles on TL and BR ------------------------------------
    tl_px = corners_pixel[0]
    br_px = corners_pixel[2]
    blue = (255, 0, 0)
    if tl_px is not None:
        tl_crop = (tl_px[0] - crop_offset_x, tl_px[1] - crop_offset_y)
        if 0 <= tl_crop[0] < orig_w and 0 <= tl_crop[1] < orig_h:
            cv2.circle(img_crop, tl_crop, 12, blue, 2)
    if br_px is not None:
        br_crop = (br_px[0] - crop_offset_x, br_px[1] - crop_offset_y)
        if 0 <= br_crop[0] < orig_w and 0 <= br_crop[1] < orig_h:
            cv2.circle(img_crop, br_crop, 12, blue, 2)

    cv2.imwrite(output_path, img_crop)
    return length_m


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolution = 155  # <- change here to check the results with different resolution in table v
    folder = os.path.join(f"{script_dir}", f"{resolution}")
    config_path = os.path.join(script_dir, "quantification", "config", "camera.json")
    tunable_path = os.path.join(script_dir, "quantification", "config", "tunable_params.json")
    camera_yaw = json.load(open(tunable_path))["camera_yaw"]
    camera_pitch = json.load(open(tunable_path))["camera_pitch"]

    with open(os.path.join(folder, "crop_coordinates.json")) as f:
        crop_data = json.load(f)
    crop_x = crop_data["top_left_x"]
    crop_y = crop_data["top_left_y"]

    mask_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

    original_img_path = os.path.join(f"{script_dir}", f"c1_{resolution}.png")
    original_img = cv2.imread(original_img_path)

    for mask_file in mask_files:
        mask_path = os.path.join(folder, mask_file)
        mask_image = cv2.imread(mask_path)
        output_viz = os.path.join(folder, f"{os.path.splitext(mask_file)[0]}_viz.png")
        length_m = visualize_crack_length(
            mask_image, crop_x, crop_y, config_path,
            original_img, output_viz, camera_yaw, camera_pitch
        )
        if length_m is not None:
            print(f"  {mask_file:<45}  {length_m} m  →  viz: {os.path.basename(output_viz)}")
        else:
            print(f"  {mask_file:<45}  FAILED (None returned)")
